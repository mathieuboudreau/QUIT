/*
 *  MultiEchoSequence.h
 *
 *  Copyright (c) 2018 Tobias Wood.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#ifndef SEQUENCES_SPINECHO_H
#define SEQUENCES_SPINECHO_H

#include "SequenceBase.h"

namespace QI {

struct MultiEchoSequence : SequenceBase {
    double         TR;
    Eigen::ArrayXd TE;
    Eigen::Index   size() const override;
    QI_SEQUENCE_DECLARE(MultiEcho);
};
void from_json(const json &j, MultiEchoSequence &s);
void to_json(json &j, const MultiEchoSequence &s);

} // End namespace QI

#endif // SEQUENCES_SPINECHO_H
