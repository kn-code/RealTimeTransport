//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/ComputePropagator.h"

namespace RealTimeTransport
{

// For t=0...model.tCrit() computes the propagator in a numerically stable way
SciCore::Cheb<SciCore::Matrix> computePropagatorMinusOneBlock(
    const SciCore::Matrix& minusILInfty,
    const SciCore::ChebAdaptive<SciCore::Matrix>& minusIK,
    SciCore::Real errorGoal,
    SciCore::Real tCrit)
{
    using namespace SciCore;

    if (minusIK.lowerLimit() != 0)
    {
        throw Error("Invalid lower limit.");
    }

    if (minusIK.upperLimit() < tCrit)
    {
        throw Error("Invalid upper limit.");
    }

    // Create Cheb interpolation of K from 0...tCrit()
    int nCheb = 64;
    Cheb<Matrix> minusIK0(minusIK, 0, tCrit, nCheb);
    Cheb<Matrix> minusIK0Integrated = minusIK0.integrate();

    Cheb h([&](Real t) -> Matrix { return minusILInfty + minusIK0Integrated(t); }, 0, tCrit, nCheb);
    Matrix zero = Matrix::Zero(minusILInfty.rows(), minusILInfty.cols());
    Cheb<Matrix> returnValue =
        SciCore::Detail::solveIdeChebMatrixValued(minusILInfty, minusIK0, h, zero, 0, tCrit, nCheb);

    // The heuristic safety factor 0.001 is needed because we want to make sure that
    // not just Π(t)-I is accurate for small t, but also (Π-I)/t !
    bool ok = returnValue.chopCoefficients(0, Real(0.001) * errorGoal);

    if (!ok)
    {
        throw Error("chopCoefficients failed.");
    }

    return returnValue;
}

BlockDiagonalCheb computePropagatorMinusOne(
    const BlockDiagonalMatrix& minusILInfty,
    const BlockDiagonalCheb& minusIK,
    SciCore::Real errorGoal,
    SciCore::Real tCrit)
{
    using namespace SciCore;

    std::vector<ChebAdaptive<Matrix>> blocks;

    int numBlocks = minusILInfty.blockDimensions().size();
    blocks.reserve(numBlocks);

    for (int i = 0; i < numBlocks; ++i)
    {
        Cheb<Matrix> PiM1 = computePropagatorMinusOneBlock(minusILInfty(i), minusIK.block(i), errorGoal, tCrit);
        blocks.emplace_back(ChebAdaptive<Matrix>(&PiM1, 1));
    }

    return BlockDiagonalCheb(std::move(blocks));
}

} // namespace RealTimeTransport
