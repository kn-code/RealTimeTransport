//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_RENORMALIZED_PT_DIAGRAMS_H
#define REAL_TIME_TRANSPORT_RENORMALIZED_PT_DIAGRAMS_H

#include "../Model.h"
#include "../Utility.h"
#include "../BlockMatrices/BlockVector.h"

#include <functional>

namespace RealTimeTransport::RenormalizedPT::Detail
{

//  _____________
//  |           |
//  |           |
//  0===========0
//
SciCore::Matrix diagram_1_regular(
    int blockIndex,
    SciCore::Real t,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::vector<BlockMatrix>& superfermions,
    const Model* model);

//  _____________
//  |           |
//  |           |
//  0===========0
//
SciCore::Matrix diagram_1_small_t(
    int blockIndex,
    SciCore::Real t,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePiMinusOne,
    const std::vector<BlockMatrix>& superfermion,
    const Model* model);

//  _____________
//  |           |
//  |           |
//  0===========0
//
SciCore::Matrix diagram_1(
    int blockIndex,
    SciCore::Real t,
    SciCore::Real tCrit,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePiMinusOne,
    const std::vector<BlockMatrix>& superfermion,
    const Model* model);

//
//       __
//  ______|______
//  |     |     |
//  |     |     |
//  0=====0=====0
//  t     τ     s
//  i2    i1  \bar{i2}
//
// Computes the column col of the above diagram with middle index i1
BlockVector effectiveVertexDiagram1_col(
    int i1,
    int col,
    SciCore::Real t_minus_tau,
    SciCore::Real tau_minus_s,
    SciCore::Real tCrit,
    SciCore::Real epsAbs,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePiMinusOne,
    const std::vector<BlockMatrix>& superfermion,
    const Model* model);

// Computes the complete representation of the effective vertex i1 if col == -1.
// Otherwise, if col >= 0, only the column col is computed.
BlockMatrix effectiveVertexDiagram1_fromCols(
    int i1,
    SciCore::Real t_minus_tau,
    SciCore::Real tau_minus_s,
    const std::function<BlockVector(int, int, SciCore::Real, SciCore::Real)>& computeD_col,
    int col,
    const Model* model);

//  _____________
//  |           |
//  |           |
//  0===========O <-- effectiveVertex
//
SciCore::Matrix diagram_2(
    int blockIndex,
    SciCore::Real t,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockVector(int, int, SciCore::Real, SciCore::Real)>& computeD_col,
    const std::vector<BlockMatrix>& superfermion,
    const Model* model);

//  _________________________
//  |       _________       |
//  |       |       |       |
//  0-------0-------0-------0
//
SciCore::Matrix diagram_2_2(
    int blockIndex,
    SciCore::Real t,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computeDiagram_1,
    const std::vector<BlockMatrix>& superfermion,
    const Model* model);

//  _____________
//  |           |
//  |           |
//  X===========0
//
Model::SuperRowVectorType currentDiagram_1(
    SciCore::Real t,
    int r,
    SciCore::Real tCrit,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePiMinusOne,
    const Model::SuperRowVectorType& idRow,
    const std::vector<Model::SuperRowVectorType>& Tr_superfermionAnnihilation,
    const std::vector<BlockMatrix>& superfermion,
    const std::vector<int>& blockStartIndices,
    const Model* model);

//  _____________
//  |           |
//  |           |
//  X===========O <-- effectiveVertex
//
Model::SuperRowVectorType currentDiagram_2(
    SciCore::Real t,
    int r,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockVector(int, int, SciCore::Real, SciCore::Real)>& computeD_col,
    const std::vector<Model::SuperRowVectorType>& Tr_superfermionAnnihilation,
    const std::vector<int>& blockStartIndices,
    int block,
    const Model* model);

//  _________________________
//  |       _________       |
//  |       |       |       |
//  X-------0-------0-------0
//
Model::SuperRowVectorType currentDiagram_2_2(
    SciCore::Real t,
    int r,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computeDiagram_1,
    const std::vector<BlockMatrix>& superfermion,
    const std::vector<Model::SuperRowVectorType>& Tr_superfermionAnnihilation,
    const std::vector<int>& blockStartIndices,
    const Model* model);

} // namespace RealTimeTransport::RenormalizedPT::Detail

#endif // REAL_TIME_TRANSPORT_RENORMALIZED_PT_DIAGRAMS_H
