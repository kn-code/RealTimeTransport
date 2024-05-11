//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#define REAL_TIME_TRANSPORT_DEBUG

#include <gtest/gtest.h>

#include <SciCore/Parallel.h>

#include <RealTimeTransport/BlockMatrices/BlockDiagonalCheb.h>
#include <RealTimeTransport/BlockMatrices/MatrixOperations.h>

using namespace SciCore;
using namespace RealTimeTransport;

TEST(BlockDiagonalCheb, BasicTest)
{
    auto f = [](int i, Real t) -> Matrix
    {
        Complex I(0, 1);
        if (i == 0)
        {
            return Matrix{
                {std::sin(t),         1.0 - t},
                {std::cos(t), std::exp(I * t)}
            };
        }
        else if (i == 1)
        {
            return Matrix{{4}};
        }
        else if (i == 2)
        {
            return Matrix{
                {std::sin(t) * t,           t - t * t,               I * t},
                {std::cos(t) * t, t * std::exp(I * t),                   0},
                {              2, std::cos(t) * t * t, std::sin(t) * t * t}
            };
        }
        else
        {
            throw Error("Invalid index");
        }
    };

    int nBlocks = 3;
    Real a      = -2;
    Real b      = 1;
    Real epsAbs = 1e-6;
    Real epsRel = 0;
    Real hMin   = 1e-4;
    bool ok     = false;
    BlockDiagonalCheb cheb(f, nBlocks, a, b, epsAbs, epsRel, hMin, &ok);

    EXPECT_TRUE(ok);

    RealVector testTimes = RealVector::LinSpaced(200, a, b);
    for (Real t : testTimes)
    {
        BlockDiagonalMatrix expected(std::vector<Matrix>{f(0, t), f(1, t), f(2, t)});
        EXPECT_LT(maxNorm(expected - cheb(t)), epsAbs);
    }

    // Compute multi threaded, check that its the same
    tf::Executor executor;
    BlockDiagonalCheb cheb2(f, nBlocks, a, b, epsAbs, epsRel, hMin, executor, &ok);
    EXPECT_EQ(cheb, cheb2);
}
