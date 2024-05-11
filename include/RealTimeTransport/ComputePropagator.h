//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_COMPUTE_PROPAGATOR_H
#define REAL_TIME_TRANSPORT_COMPUTE_PROPAGATOR_H

#include <SciCore/IDECheb.h>

#include "BlockMatrices/BlockDiagonalCheb.h"
#include "Propagator.h"

namespace RealTimeTransport
{

template <typename MemoryKernelT>
SciCore::ChebAdaptive<SciCore::Matrix> computePropagatorBlock(const MemoryKernelT& memoryKernel, int blockIndex)
{
    using namespace SciCore;

    Real t0      = 0;
    Real tMax    = memoryKernel.tMax();
    Real epsAbs  = memoryKernel.errorGoal();
    Real epsRel  = 0;
    int nMinCheb = 3 * 16;

    return computePropagatorIde(
        memoryKernel.LInfty()(blockIndex), memoryKernel.K().block(blockIndex), t0, tMax, epsAbs, epsRel, nMinCheb);
}

template <typename MemoryKernelT>
BlockDiagonalCheb computePropagatorTemplate(const MemoryKernelT& memoryKernel, int block)
{
    using namespace SciCore;

    std::vector<ChebAdaptive<Matrix>> blocks;

    int numBlocks = memoryKernel.model()->blockDimensions().size();
    blocks.reserve(numBlocks);

    // Compute all blocks
    if (block < 0)
    {
        for (int i = 0; i < numBlocks; ++i)
        {
            blocks.emplace_back(computePropagatorBlock(memoryKernel, i));
        }
    }
    // Compute only one specific block, set all other blocks to zero
    else
    {
        for (int i = 0; i < numBlocks; ++i)
        {
            if (i == block)
            {
                blocks.emplace_back(computePropagatorBlock(memoryKernel, i));
            }
            else
            {
                blocks.emplace_back(ChebAdaptive<Matrix>(
                    [&](Real) -> Matrix {
                        return Matrix::Zero(
                            memoryKernel.model()->blockDimensions()[i], memoryKernel.model()->blockDimensions()[i]);
                    },
                    0.0, memoryKernel.tMax(), memoryKernel.errorGoal(), 0.0, 0.0));
            }
        }
    }

    return BlockDiagonalCheb(std::move(blocks));
}

// For t=0...model.tCrit() computes the propagator in a numerically stable way
SciCore::Cheb<SciCore::Matrix> computePropagatorMinusOneBlock(
    const SciCore::Matrix& minusILInfty,
    const SciCore::ChebAdaptive<SciCore::Matrix>& minusIK,
    SciCore::Real errorGoal,
    SciCore::Real tCrit);

BlockDiagonalCheb computePropagatorMinusOne(
    const BlockDiagonalMatrix& minusILInfty,
    const BlockDiagonalCheb& minusIK,
    SciCore::Real errorGoal,
    SciCore::Real tCrit);

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_COMPUTE_PROPAGATOR_H
