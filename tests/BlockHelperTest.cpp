//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#define REAL_TIME_TRANSPORT_DEBUG

#include <gtest/gtest.h>

#include <RealTimeTransport/BlockMatrices/BlockHelper.h>

using namespace SciCore;
using namespace RealTimeTransport;

TEST(extractDenseBlock, BasicTest)
{
    // Initialize a test matrix (5x5 for simplicity)
    RealMatrix A{
        { 1,  2,  3,  4,  5},
        { 6,  7,  8,  9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25}
    };

    // Block dimensions (two 2x2 blocks and one 1x1 block)
    std::vector<int> blockDims = {2, 2, 1};

    // Extracted blocks
    RealMatrix block00 = extractDenseBlock(A, 0, 0, blockDims);
    RealMatrix block01 = extractDenseBlock(A, 0, 1, blockDims);
    RealMatrix block02 = extractDenseBlock(A, 0, 2, blockDims);
    RealMatrix block10 = extractDenseBlock(A, 1, 0, blockDims);
    RealMatrix block11 = extractDenseBlock(A, 1, 1, blockDims);
    RealMatrix block12 = extractDenseBlock(A, 1, 2, blockDims);
    RealMatrix block20 = extractDenseBlock(A, 2, 0, blockDims);
    RealMatrix block21 = extractDenseBlock(A, 2, 1, blockDims);
    RealMatrix block22 = extractDenseBlock(A, 2, 2, blockDims);

    // Expected blocks
    RealMatrix expected_00{
        {1, 2},
        {6, 7}
    };

    RealMatrix expected_01{
        {3, 4},
        {8, 9}
    };

    RealMatrix expected_02{{5}, {10}};

    RealMatrix expected_10{
        {11, 12},
        {16, 17}
    };

    RealMatrix expected_11{
        {13, 14},
        {18, 19}
    };

    RealMatrix expected_12{{15}, {20}};

    RealMatrix expected_20{
        {21, 22}
    };

    RealMatrix expected_21{
        {23, 24},
    };

    RealMatrix expected_22{{25}};

    // Check if the extracted block matches the expected block
    EXPECT_EQ(block00, expected_00);
    EXPECT_EQ(block01, expected_01);
    EXPECT_EQ(block02, expected_02);
    EXPECT_EQ(block10, expected_10);
    EXPECT_EQ(block11, expected_11);
    EXPECT_EQ(block12, expected_12);
    EXPECT_EQ(block20, expected_20);
    EXPECT_EQ(block21, expected_21);
    EXPECT_EQ(block22, expected_22);
}

TEST(isBlockDiagonal, Test)
{
    // Test case 1: A block diagonal matrix
    RealMatrix mat1{
        {1, 0, 0, 0, 0,  0},
        {0, 2, 0, 0, 0,  0},
        {0, 0, 3, 4, 2,  0},
        {0, 0, 5, 6, 1,  0},
        {0, 0, 4, 5, 7,  0},
        {0, 0, 0, 0, 0, 10}
    };
    std::vector<int> blocks1 = {2, 3, 1};
    EXPECT_TRUE(isBlockDiagonal(mat1, blocks1));

    // Test case 2: Not a block diagonal matrix (has non-zero elements outside blocks)
    RealMatrix mat2{
        {1, 0, 0, 1},
        {0, 2, 0, 0},
        {0, 0, 3, 0},
        {0, 0, 0, 4}
    };
    std::vector<int> blocks2 = {2, 2};
    EXPECT_FALSE(isBlockDiagonal(mat2, blocks2));

    // Test case 3: A single block matrix (should return true)
    RealMatrix mat3{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    std::vector<int> blocks3 = {3};
    EXPECT_TRUE(isBlockDiagonal(mat3, blocks3));

    // Test case 4: Empty matrix (edge case)
    Eigen::MatrixXd mat4(0, 0);
    std::vector<int> blocks4 = {};
    EXPECT_TRUE(isBlockDiagonal(mat4, blocks4));
}
