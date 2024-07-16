//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_RENORMALIZED_PT_CONDUCTANCE_DIAGRAMS_H
#define REAL_TIME_TRANSPORT_RENORMALIZED_PT_CONDUCTANCE_DIAGRAMS_H

#include "../BlockMatrices/BlockDiagonalCheb.h"
#include "../BlockMatrices/BlockDiagonalMatrix.h"
#include "../BlockMatrices/BlockMatrix.h"
#include "../BlockMatrices/BlockVector.h"
#include "../Model.h"

#include <SciCore/ChebAdaptive.h>
#include <SciCore/Utility.h>

#include <Eigen/QR>

#include <functional>

namespace RealTimeTransport::RenormalizedPT::Detail
{

//
// Compute dρ/dμ by solving dΣ/dμ ρ + Σ dρ/dμ = 0, where ρ denotes the stationary state
// and Σ the memory kernel at zero-freqeuncy. It must be enforced that Tr dρ/dμ = 0.
//
template <typename MemoryKernelT>
Model::SupervectorType compute_d_dmu_rhoStat(
    const MemoryKernelT& memoryKernel,
    const BlockDiagonalCheb& d_dmu_memoryKernel,
    const Model::SupervectorType& rhoStat,
    const Model::SuperRowVectorType& idRow,
    int block)
{
    using namespace SciCore;
    using Supervector = Model::SupervectorType;

    Supervector returnValue           = Supervector::Zero(idRow.size());
    Real tMax                         = memoryKernel.tMax();
    const std::vector<int>& blockDims = memoryKernel.model()->blockDimensions();

    // FIXME this only uses the block structure if block==0, otherwise the full equation is solved
    if (block < 0 || block > 0)
    {
        BlockDiagonalMatrix d_dmu_SigmaZeroFreq = d_dmu_memoryKernel.integrate()(tMax);
        Supervector b                           = -(d_dmu_SigmaZeroFreq * rhoStat);
        Matrix A                                = memoryKernel.zeroFrequency().toDense();
        A.row(A.rows() - 1)                     = idRow;
        b[A.rows() - 1]                         = 0.0;
        returnValue                             = A.colPivHouseholderQr().solve(b);
    }
    else if (block == 0)
    {
        if (rhoStat.tail(rhoStat.size() - blockDims[block]).isZero() == false)
        {
            throw Error("Stationary state does not have the required structure");
        }

        if (idRow.tail(idRow.size() - blockDims[block]).isZero() == false)
        {
            throw Error("Inconsistent block structure");
        }

        Matrix A                   = memoryKernel.zeroFrequency()(block);
        Matrix d_dmu_SigmaZeroFreq = d_dmu_memoryKernel.block(block).integrate()(tMax);
        Supervector b              = -(d_dmu_SigmaZeroFreq * rhoStat.segment(0, blockDims[block]));

        A.row(A.rows() - 1)                = idRow.segment(0, blockDims[block]);
        b[A.rows() - 1]                    = 0.0;
        returnValue.head(blockDims[block]) = A.colPivHouseholderQr().solve(b);
    }
    else
    {
        throw Error("Logic error");
    }

    truncToZero(returnValue, std::numeric_limits<Real>::epsilon());
    return returnValue;
}

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
