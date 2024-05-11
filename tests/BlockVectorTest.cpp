//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#define REAL_TIME_TRANSPORT_DEBUG

#include <gtest/gtest.h>

#include <RealTimeTransport/BlockMatrices/BlockVector.h>

using namespace SciCore;
using namespace RealTimeTransport;

TEST(BlockVector, operatorPlus)
{
    BlockVector<Complex> x;
    x.emplace(
        0, Matrix{
               { 1, 4},
               {-1, 2}
    });
    x.emplace(
        1, Matrix{
               {5, 6},
               {7, 8}
    });
    x.emplace(
        4, Matrix{
               {1, 2},
               {3, 4}
    });

    BlockVector<Complex> y;
    y.emplace(
        0, Matrix{
               {2, 8},
               {2, 9}
    });
    y.emplace(
        1, Matrix{
               {-5, -6},
               {-7, -8}
    });
    y.emplace(
        3, Matrix{
               {1, 2},
               {3, 4}
    });

    x += y;

    ASSERT_TRUE(x.contains(0));
    ASSERT_TRUE(x.contains(1));
    ASSERT_TRUE(x.contains(2) == false);
    ASSERT_TRUE(x.contains(3));
    ASSERT_TRUE(x.contains(4));

    Matrix result0{
        {3, 12},
        {1, 11}
    };

    Matrix result1{
        {0, 0},
        {0, 0}
    };

    Matrix result3{
        {1, 2},
        {3, 4}
    };

    Matrix result4{
        {1, 2},
        {3, 4}
    };

    EXPECT_TRUE(x.find(0)->second.isApprox(result0));
    EXPECT_TRUE(x.find(1)->second.isApprox(result1));
    EXPECT_TRUE(x.find(3)->second.isApprox(result3));
    EXPECT_TRUE(x.find(4)->second.isApprox(result4));

    x.eraseZeroes();
    ASSERT_TRUE(x.contains(1) == false);
}
