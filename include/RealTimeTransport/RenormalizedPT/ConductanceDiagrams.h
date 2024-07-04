//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_RENORMALIZED_PT_CONDUCTANCE_DIAGRAMS_H
#define REAL_TIME_TRANSPORT_RENORMALIZED_PT_CONDUCTANCE_DIAGRAMS_H

#include "../BlockMatrices/BlockDiagonalMatrix.h"
#include "../BlockMatrices/BlockMatrix.h"
#include "../BlockMatrices/BlockVector.h"
#include "../Model.h"

#include <functional>

namespace RealTimeTransport::RenormalizedPT::Detail
{

//  ______/______
//  |           |
//  |           |
//  0===========0
//
SciCore::Matrix d_dmu_diagram_1(
    int blockIndex,
    SciCore::Real t,
    int r,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::vector<BlockMatrix>& superfermion,
    const Model* model);

//
//       __
//  ___/__|______
//  |     |     |
//  |     |     |
//  0=====0=====0
//  t     τ     s
//  i2    i1  \bar{i2}
//
// Computes the column col of the above diagram with middle index i1
BlockVector d_dmu_effectiveVertexDiagram1_col(
    int i1,
    int col,
    SciCore::Real t_minus_tau,
    SciCore::Real tau_minus_s,
    int r,
    SciCore::Real epsAbs,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::vector<BlockMatrix>& superfermion,
    const Model* model);

//  ______/______
//  |           |
//  |           |
//  0===========O <-- effectiveVertex
//
//  +
//  _____________
//  |           |
//  |           |
//  0===========d/dμ(O) <-- derivative effectiveVertex
//
SciCore::Matrix d_dmu_diagram_2(
    int blockIndex,
    SciCore::Real t,
    int r,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockVector(int, int, SciCore::Real, SciCore::Real)>& computeD_col,
    const std::function<BlockVector(int, int, SciCore::Real, SciCore::Real, int)>& compute_d_dmu_D_col,
    const std::vector<BlockMatrix>& superfermion,
    const Model* model);

//  ____________/____________
//  |       _________       |
//  |       |       |       |
//  0-------0-------0-------0
//
//  +
//  _________________________
//  |       ____/____       |
//  |       |       |       |
//  0-------0-------0-------0
//
SciCore::Matrix d_dmu_diagram_2_2(
    int blockIndex,
    SciCore::Real t,
    int r,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computeDiagram_1,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& compute_d_dmu_diagram_1,
    const std::vector<BlockMatrix>& superfermion,
    const Model* model);

//  ______/______
//  |           |
//  |           |
//  X===========0
//
Model::SuperRowVectorType d_dmu_currentDiagram_1(
    SciCore::Real t,
    int rI,
    int rmu,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::vector<Model::SuperRowVectorType>& Tr_superfermionAnnihilation,
    const std::vector<BlockMatrix>& superfermion,
    const std::vector<int>& blockStartIndices,
    const Model* model);

//  ______/______
//  |           |
//  |           |
//  X===========O <-- effectiveVertex
//
//  +
//  _____________
//  |           |
//  |           |
//  X===========d/dμ(O) <-- derivative effectiveVertex
//
Model::SuperRowVectorType d_dmu_currentDiagram_2(
    SciCore::Real t,
    int rI,
    int rmu,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockVector(int, int, SciCore::Real, SciCore::Real)>& computeD_col,
    const std::function<BlockVector(int, int, SciCore::Real, SciCore::Real, int)>& compute_d_dmu_D_col,
    const std::vector<Model::SuperRowVectorType>& Tr_superfermionAnnihilation,
    const std::vector<int>& blockStartIndices,
    int block,
    const Model* model);

//  ____________/____________
//  |       _________       |
//  |       |       |       |
//  X-------0-------0-------0
//
//  +
//  _________________________
//  |       ____/____       |
//  |       |       |       |
//  X-------0-------0-------0
//
Model::SuperRowVectorType d_dmu_currentDiagram_2_2(
    SciCore::Real t,
    int rI,
    int rmu,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computeDiagram_1,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& compute_d_dmu_diagram_1,
    const std::vector<BlockMatrix>& superfermion,
    const std::vector<Model::SuperRowVectorType>& Tr_superfermionAnnihilation,
    const std::vector<int>& blockStartIndices,
    const Model* model);

} // namespace RealTimeTransport::RenormalizedPT::Detail

#endif // REAL_TIME_TRANSPORT_RENORMALIZED_PT_CONDUCTANCE_DIAGRAMS_H
