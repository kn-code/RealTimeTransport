//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_ITERATED_RG_CONDUCTANCE_DIAGRAMS_H
#define REAL_TIME_TRANSPORT_ITERATED_RG_CONDUCTANCE_DIAGRAMS_H

#include "../BlockMatrices/BlockDiagonalMatrix.h"
#include "../BlockMatrices/BlockMatrix.h"
#include "../Model.h"

#include <functional>

namespace RealTimeTransport::IteratedRG::Detail
{

//  _________________________
//  |       ____/____       |
//  |       |       |       |
//  0-------0-------0-------0
//
SciCore::Matrix d_dmu_diagram_2_2(
    int blockIndex,
    SciCore::Real t,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& compute_d_dmu_diagram_1,
    const std::vector<BlockMatrix>& superfermion,
    const Model* model);

//  _________________________
//  |       ____/____       |
//  |       |       |       |
//  X-------0-------0-------0
//
Model::SuperRowVectorType d_dmu_currentDiagram_2_2(
    SciCore::Real t,
    int rI,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& compute_d_dmu_diagram_1,
    const std::vector<BlockMatrix>& superfermion,
    const std::vector<Model::SuperRowVectorType>& Tr_superfermionAnnihilation,
    const std::vector<int>& blockStartIndices,
    const Model* model);

} // namespace RealTimeTransport::IteratedRG::Detail

#endif // REAL_TIME_TRANSPORT_ITERATED_RG_CONDUCTANCE_DIAGRAMS_H
