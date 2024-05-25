//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_ITERATED_RG_DIAGRAMS_H
#define REAL_TIME_TRANSPORT_ITERATED_RG_DIAGRAMS_H

#include <functional>

#include "../BlockMatrices/BlockDiagonalMatrix.h"
#include "../BlockMatrices/BlockVector.h"
#include "../Model.h"

namespace RealTimeTransport::IteratedRG::Detail
{

//
//       __    __
//  ______|_____|______
//  |     |     |     |
//  |     |     |     |
//  0=====0=====0=====0
//  t    τ_1   τ_2    s
//  i3    i1    i2  \bar{i3}
//
// Computes column col of the above diagram
BlockVector bareTwoPointVertex_col(
    int i1,
    int i2,
    int col,
    SciCore::Real t_minus_tau1,
    SciCore::Real tau1_minus_tau2,
    SciCore::Real tau2_minus_s,
    const BlockDiagonalMatrix& Pi1,
    const BlockDiagonalMatrix& Pi2,
    const BlockDiagonalMatrix& Pi3,
    SciCore::Real epsAbs,
    const std::vector<Model::SuperfermionType>& superfermion,
    const Model* model);

//
//       __
//  ______|______
//  |     |     |
//  |     |     |
//  0=====0=====E  <-- effective vertex O(D^3)
//  t     τ
//  i2    i1  \bar{i2}
//
// Computes the column col of the above diagram with middle index i1
BlockVector effectiveVertexCorrection1_col(
    int i1,
    int col,
    SciCore::Real t_minus_tau,
    SciCore::Real tau_minus_s,
    SciCore::Real epsAbs,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockVector(int, int, SciCore::Real, SciCore::Real)>& computeD_col,
    const std::vector<Model::SuperfermionType>& superfermion,
    const Model* model);

//
//       __
//  ______|______
//  |     |     |
//  |     |     |
//  0=====E=====0  ( E is effective vertex O(D^3) )
//  t     τ
//  i2    i1  \bar{i2}
//
// Computes the column col of the above diagram with middle index i1
BlockVector effectiveVertexCorrection2_col(
    int i1,
    int col,
    SciCore::Real t_minus_tau,
    SciCore::Real tau,
    SciCore::Real epsAbs,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockVector(int, int, SciCore::Real, SciCore::Real)>& computeD_col,
    const std::vector<Model::SuperfermionType>& superfermion,
    const Model* model);

//
//              __
//  _____________|
//  |           ||
//  |           ||
//  0===========00
//  t            τ
//  i2        \bar{i2},i1
//
BlockVector effectiveVertexCorrection3_col(
    int i1,
    int col,
    SciCore::Real t_minus_tau,
    SciCore::Real tau,
    SciCore::Real epsAbs,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::vector<Model::SuperfermionType>& superfermion,
    const Model* model);

//
//             __
//  ____________|_
//  |           ||
//  |           ||
//  0===========00
//  t           τ
//  i2        i1,\bar{i2}
//
BlockVector effectiveVertexCorrection4_col(
    int i1,
    int col,
    SciCore::Real t_minus_tau,
    SciCore::Real tau,
    SciCore::Real epsAbs,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::vector<Model::SuperfermionType>& superfermion,
    const Model* model);

} // namespace RealTimeTransport::IteratedRG::Detail

#endif // REAL_TIME_TRANSPORT_ITERATED_RG_DIAGRAMS_H
