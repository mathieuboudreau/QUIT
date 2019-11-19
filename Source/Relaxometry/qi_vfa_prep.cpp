/*
 *  qi_vfa_prep.cpp
 *
 *  Copyright (c) 2019 Tobias Wood.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <type_traits>
#include <unsupported/Eigen/MatrixFunctions>

// #define QI_DEBUG_BUILD 1

#include "Args.h"
#include "ImageIO.h"
#include "Macro.h"
#include "Model.h"
#include "ModelFitFilter.h"
#include "SequenceBase.h"
#include "SimulateModel.h"
#include "Util.h"

struct VFAPrepSequence : QI::SequenceBase {
    double         TE, TR, FA;
    int            SPS;
    Eigen::ArrayXd PrepFA;
    QI_SEQUENCE_DECLARE(VFAPrep);
    Eigen::Index size() const override { return PrepFA.size(); };
};
void from_json(const json &j, VFAPrepSequence &s) {
    j.at("TE").get_to(s.TE);
    j.at("TR").get_to(s.TR);
    j.at("SPS").get_to(s.SPS);
    s.FA     = j.at("FA").get<double>() * M_PI / 180.0;
    s.PrepFA = QI::ArrayFromJSON(j, "PrepFA", M_PI / 180.0);
}

void to_json(json &j, const VFAPrepSequence &s) {
    j = json{{"TE", s.TE},
             {"TR", s.TR},
             {"SPS", s.SPS},
             {"FA", s.FA * 180 / M_PI},
             {"PrepFA", s.PrepFA * 180 / M_PI}};
}

struct VFAPrepModel {
    using DataType      = double;
    using ParameterType = double;

    static constexpr int NV = 4; // Number of varying parameters
    static constexpr int ND = 0; // Number of derived parameters
    static constexpr int NF = 0; // Number of fixed parameters
    static constexpr int NI = 1; // Number of inputs

    using VaryingArray = QI_ARRAYN(ParameterType, NV); // Type for the varying parameter array
    using FixedArray   = QI_ARRAYN(ParameterType, NF); // Type for the fixed parameter array

    // Sequence paramter structs
    VFAPrepSequence &sequence;

    // Fitting start point and bounds
    // The PD will be scaled by the fitting function to keep parameters roughly the same magnitude
    VaryingArray const start{30., 1., 0.05, 1.0};
    VaryingArray const bounds_lo{1, 0.7, 0.01, 0.5};
    VaryingArray const bounds_hi{150, 1.5, 3, 1.5};

    std::array<std::string, NV> const varying_names{"PD", "T1", "T2", "B1"};
    std::array<std::string, NF> const fixed_names{};
    // If fixed parameters not supplied, use these default values
    FixedArray const fixed_defaults{};

    template <typename Derived>
    auto signal(Eigen::ArrayBase<Derived> const &v, FixedArray const &) const
        -> QI_ARRAY(typename Derived::Scalar) {
        using T     = typename Derived::Scalar;
        using Mat44 = Eigen::Matrix<T, 4, 4>;
        T const &PD = v[0];
        T const &R1 = 1. / v[1];
        T const &R2 = 1. / v[2];
        T const &B1 = v[3];

        auto const Relax = [&PD, &R1, &R2](double const t) {
            Mat44 R;
            R << -R2, 0, 0, 0,      //
                0, -R2, 0, 0,       //
                0, 0, -R1, PD * R1, //
                0, 0, 0, 0;
            Mat44 eRt = (R * t).exp();
            return eRt;
        };

        auto const RF = [&](double const a, double const ph) {
            auto const      ca = cos(B1 * a);
            auto const      sa = sin(B1 * a);
            auto const      ux = cos(ph);
            auto const      uy = sin(ph);
            Eigen::Matrix4d A;
            A << ca + ux * ux * (1 - ca), ux * uy * (1 - ca), -uy * sa, 0., //
                ux * uy * (1 - ca), ca + uy * uy * (1 - ca), ux * sa, 0.,   //
                uy * sa, -ux * sa, ca, 0.,                                  //
                0., 0., 0., 1.;
            return A;
        };
        auto const A     = RF(sequence.FA, 0);
        auto const R     = Relax(sequence.TR);
        auto const S     = Eigen::DiagonalMatrix<double, 4, 4>({0, 0, 1., 1.}).toDenseMatrix();
        auto const RUFIS = A * S * R;
        auto const seg2  = RUFIS.pow(sequence.SPS / 2.0);

        auto const gap  = Relax(sequence.TE / 4);
        auto const ref1 = RF(M_PI, M_PI / 2);
        auto const ref2 = RF(-M_PI, M_PI / 2);
        QI_ARRAY(T) signal(sequence.size());
        for (long i = 0; i < sequence.size(); i++) {
            auto const tip_down = RF(sequence.PrepFA[i], 0);
            auto const tip_up   = RF(-sequence.PrepFA[i], 0);
            auto const prep     = tip_up * gap * ref2 * gap * gap * ref1 * gap * tip_down;

            auto const                X = seg2 * prep * seg2;
            Eigen::EigenSolver<Mat44> es(X);
            long                      index;
            (es.eigenvalues().array().abs() - 1.0).abs().minCoeff(&index);
            auto const M = es.eigenvectors().col(index).array().abs();
            signal[i]    = sqrt(M[0] * M[0] + M[1] * M[1]) / M[3];

            // QI_DBMAT(seg2);
            // QI_DBMAT(prep);
            // QI_DBMAT(X);
            // QI_DBMAT(es.eigenvectors());
            // QI_DBVEC(es.eigenvalues());
            // QI_DB(index);
            // QI_DBVEC(M);
        }
        QI_DB(PD);
        QI_DB(1 / R1);
        QI_DB(1 / R2);
        QI_DBVEC(signal);
        return signal;
    }
};

struct VFAPrepCost {
    VFAPrepModel const &     model;
    VFAPrepModel::FixedArray fixed;
    QI_ARRAY(double) const data;

    template <typename T> bool operator()(T const *const vin, T *rin) const {
        Eigen::Map<QI_ARRAYN(T, VFAPrepModel::NV) const> const varying(vin);
        Eigen::Map<QI_ARRAY(T)>                                residuals(rin, data.rows());
        residuals = data - model.signal(varying, fixed);
        QI_DBVEC(residuals);
        return true;
    }
};

struct VFAPrepFit {
    // Boilerplate information required by ModelFitFilter
    static const bool Blocked = false; // = input is in blocks and outputs have multiple entries
    static const bool Indexed = false; // = the voxel index will be passed to the fit
    using RMSErrorType        = double;
    using FlagType            = int; // Almost always the number of iterations

    using ModelType = VFAPrepModel;
    ModelType model;

    // Have to tell the ModelFitFilter how many volumes we expect in each input
    int input_size(const int) const { return model.sequence.size(); }

    // This has to match the function signature that will be called in ModelFitFilter (which depends
    // on Blocked/Indexed. The return type is a simple struct indicating success, and on failure
    // also the reason for failure
    QI::FitReturnType
    fit(std::vector<Eigen::ArrayXd> const &inputs,    // Input: signal data
        Eigen::ArrayXd const &             fixed,     // Input: Fixed parameters
        ModelType::VaryingArray &          varying,   // Output: Varying parameters
        RMSErrorType &                     rmse,      // Output: root-mean-square error
        std::vector<Eigen::ArrayXd> &      residuals, // Optional output: point residuals
        FlagType &                         iterations /* Usually iterations */) const {
        // First scale down the raw data so that PD will be roughly the same magnitude as other
        // parameters This is important for numerical stability in the optimiser

        QI_DBVEC(inputs[0]);

        double scale = inputs[0].maxCoeff();
        QI_DB(scale);
        if (scale < std::numeric_limits<double>::epsilon()) {
            varying = ModelType::VaryingArray::Zero();
            rmse    = 0.0;
            return {false, "Maximum data value was zero or less"};
        }
        Eigen::ArrayXd const data = inputs[0] / scale;

        // Setup Ceres
        ceres::Problem problem;
        using VFADiff = ceres::
            NumericDiffCostFunction<VFAPrepCost, ceres::CENTRAL, ceres::DYNAMIC, ModelType::NV>;
        auto *vfa_prep_cost = new VFADiff(
            new VFAPrepCost{model, fixed, data}, ceres::TAKE_OWNERSHIP, model.sequence.size());
        ceres::LossFunction *loss = new ceres::HuberLoss(1.0); // Don't know if this helps
        // This is where the parameters and cost functions actually get added to Ceres
        problem.AddResidualBlock(vfa_prep_cost, loss, varying.data());

        // Set up parameter bounds
        for (int i = 0; i < ModelType::NV; i++) {
            problem.SetParameterLowerBound(varying.data(), i, model.bounds_lo[i]);
            problem.SetParameterUpperBound(varying.data(), i, model.bounds_hi[i]);
        }

        ceres::Solver::Options options;
        ceres::Solver::Summary summary;
        options.max_num_iterations  = 30;
        options.function_tolerance  = 1e-5;
        options.gradient_tolerance  = 1e-6;
        options.parameter_tolerance = 1e-4;
        options.logging_type        = ceres::SILENT;

        varying = model.start;
        ceres::Solve(options, &problem, &summary);
        if (!summary.IsSolutionUsable()) {
            return {false, summary.FullReport()};
        }

        Eigen::ArrayXd const residual = (data - model.signal(varying, fixed)) * scale;
        rmse                          = sqrt(residual.square().mean());
        if (residuals.size() > 0) {
            residuals[0] = residual;
        }
        varying[0] *= scale; // Multiply signals/proton density back up
        QI_DBVEC(varying);
        iterations = summary.iterations.size();
        return {true, ""};
    }
};

/*
 * Main
 */
int vfa_prep_main(int argc, char **argv) {
    Eigen::initParallel();
    args::ArgumentParser          parser("Calculates T1/T2 from VFA-Prep data "
                                "data.\nhttp://github.com/spinicist/QUIT");
    args::Positional<std::string> input_path(parser, "INPUT", "Input VFA-Prep file");

    QI_COMMON_ARGS;

    QI::ParseArgs(parser, argc, argv, verbose, threads);

    QI::CheckPos(input_path);

    QI::Log(verbose, "Reading sequence parameters");
    json doc = json_file ? QI::ReadJSON(json_file.Get()) : QI::ReadJSON(std::cin);

    VFAPrepSequence sequence(doc["VFAPrep"]);
    VFAPrepModel    model{sequence};
    if (simulate) {
        QI::SimulateModel<VFAPrepModel, false>(
            doc, model, {}, {input_path.Get()}, verbose, simulate.Get());
    } else {
        VFAPrepFit fit{model};
        auto       fit_filter =
            QI::ModelFitFilter<VFAPrepFit>::New(&fit, verbose, resids, subregion.Get());
        fit_filter->ReadInputs({input_path.Get()}, {}, mask.Get());
        fit_filter->Update();
        fit_filter->WriteOutputs(prefix.Get() + "VFAPrep_");
    }
    QI::Log(verbose, "Finished.");
    return EXIT_SUCCESS;
}
