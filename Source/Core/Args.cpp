/*
 *  Args.cpp
 *
 *  Copyright (c) 2019 Tobias Wood.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#include "Args.h"
#include "Util.h"
#include "itkMultiThreaderBase.h"
#include <sstream>

namespace QI {

void ParseArgs(args::ArgumentParser &parser, int argc, char **argv, const args::Flag &verbose) {
    std::stringstream temp;
    parser.Help(temp);
    try {
        parser.ParseCLI(argc, argv);
        QI::Info(verbose, "Starting {} {}", argv[0], QI::GetVersion());
    } catch (args::Help) {
        fmt::print("{}\n", temp.str());
        exit(EXIT_SUCCESS);
    } catch (args::ParseError e) {
        QI::Fail("{}\n{}", temp.str(), e.what());
    } catch (args::ValidationError e) {
        QI::Fail("{}\n{}", temp.str(), e.what());
    }
}

void ParseArgs(args::ArgumentParser &parser,
               int                   argc,
               char **               argv,
               const args::Flag &    verbose,
               args::ValueFlag<int> &threads) {
    std::stringstream temp;
    parser.Help(temp);
    try {
        parser.ParseCLI(argc, argv);
        QI::Info(verbose, "Starting {} {}", argv[0], QI::GetVersion());
        QI::Log(verbose, "Max threads = {}", threads.Get());
        itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(threads.Get());
    } catch (args::Help) {
        fmt::print("{}\n", temp.str());
        exit(EXIT_SUCCESS);
    } catch (args::ParseError e) {
        QI::Fail("{}\n{}", temp.str(), e.what());
    } catch (args::ValidationError e) {
        QI::Fail("{}\n{}", temp.str(), e.what());
    }
}

} // End namespace QI
