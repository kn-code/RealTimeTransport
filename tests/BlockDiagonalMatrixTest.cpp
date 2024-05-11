//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#define REAL_TIME_TRANSPORT_DEBUG

#include <gtest/gtest.h>

#include <RealTimeTransport/BlockMatrices/BlockDiagonalMatrix.h>

using namespace SciCore;
using namespace RealTimeTransport;

TEST(BlockDiagonalMatrix, BasicTest)
{
    std::vector<Matrix> blocks = {
        Matrix::Constant(2, 2, 1.0), // First block of 1s
        Matrix::Constant(3, 3, 2.0)  // Second block of 2s
    };
    BlockDiagonalMatrix matrix(std::move(blocks));
    EXPECT_EQ(matrix.totalRows(), 5);
    EXPECT_EQ(matrix.totalCols(), 5);

    Matrix toDenseResult{
        {1, 1, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {0, 0, 2, 2, 2},
        {0, 0, 2, 2, 2},
        {0, 0, 2, 2, 2}
    };
    EXPECT_EQ(matrix.toDense(), toDenseResult);

    // Test elements within the first block
    EXPECT_EQ(matrix.element(0, 0), 1.0);
    EXPECT_EQ(matrix.element(0, 1), 1.0);
    EXPECT_EQ(matrix.element(1, 0), 1.0);
    EXPECT_EQ(matrix.element(1, 1), 1.0);

    // Test elements within the second block
    EXPECT_EQ(matrix.element(2, 2), 2.0);
    EXPECT_EQ(matrix.element(2, 3), 2.0);
    EXPECT_EQ(matrix.element(2, 4), 2.0);
    EXPECT_EQ(matrix.element(3, 2), 2.0);
    EXPECT_EQ(matrix.element(3, 3), 2.0);
    EXPECT_EQ(matrix.element(3, 4), 2.0);
    EXPECT_EQ(matrix.element(4, 2), 2.0);
    EXPECT_EQ(matrix.element(4, 3), 2.0);
    EXPECT_EQ(matrix.element(4, 4), 2.0);

    // Test elements outside
    EXPECT_ANY_THROW(matrix.element(0, 2));
    EXPECT_ANY_THROW(matrix.element(0, 3));
    EXPECT_ANY_THROW(matrix.element(0, 4));
    EXPECT_ANY_THROW(matrix.element(1, 2));
    EXPECT_ANY_THROW(matrix.element(1, 3));
    EXPECT_ANY_THROW(matrix.element(1, 4));
    EXPECT_ANY_THROW(matrix.element(2, 0));
    EXPECT_ANY_THROW(matrix.element(2, 1));
    EXPECT_ANY_THROW(matrix.element(3, 0));
    EXPECT_ANY_THROW(matrix.element(3, 1));
    EXPECT_ANY_THROW(matrix.element(4, 0));
    EXPECT_ANY_THROW(matrix.element(4, 1));

    // Change elements and check again
    matrix.element(0, 0) = 4.0;
    matrix.element(0, 1) = 5.0;
    matrix.element(1, 0) = 6.0;
    matrix.element(1, 1) = 7.0;
    EXPECT_EQ(matrix.element(0, 0), 4.0);
    EXPECT_EQ(matrix.element(0, 1), 5.0);
    EXPECT_EQ(matrix.element(1, 0), 6.0);
    EXPECT_EQ(matrix.element(1, 1), 7.0);

    matrix.element(2, 2) = 8.0;
    matrix.element(3, 4) = 9.0;
    matrix.element(4, 4) = 10.0;
    EXPECT_EQ(matrix.element(2, 2), 8.0);
    EXPECT_EQ(matrix.element(3, 4), 9.0);
    EXPECT_EQ(matrix.element(4, 4), 10.0);

    toDenseResult = Matrix{
        {4, 5, 0, 0,  0},
        {6, 7, 0, 0,  0},
        {0, 0, 8, 2,  2},
        {0, 0, 2, 2,  9},
        {0, 0, 2, 2, 10}
    };
    EXPECT_EQ(matrix.toDense(), toDenseResult);
}

TEST(addProduct, BlockDiagonal_x_Vector)
{
    Matrix denseA{
        { 1, 2, 0,  0, 0, 0},
        {-1, 2, 0,  0, 0, 0},
        { 0, 0, 3, -5, 9, 0},
        { 0, 0, 2, -2, 2, 0},
        { 0, 0, 2,  1, 1, 0},
        { 0, 0, 0,  0, 0, 6}
    };

    const Vector x{
        {1, 2, 3, 4, 5, 6}
    };

    Vector denseResult{
        {1, -2, 3, -4, 5, -6}
    };
    denseResult += 2.0 * denseA * x;

    BlockDiagonalMatrix A(denseA, std::vector<int>{2, 3, 1});

    Vector blockResult{
        {1, -2, 3, -4, 5, -6}
    };

    addProduct(2.0, A, x, blockResult);

    EXPECT_TRUE(denseResult.isApprox(blockResult))
        << "denseResult = " << denseResult.transpose() << "\nblockResult = " << blockResult.transpose();

    denseResult = 2.0 * denseA * x;
    blockResult = product(2.0, A, x);

    EXPECT_TRUE(denseResult.isApprox(blockResult))
        << "denseResult = " << denseResult.transpose() << "\nblockResult = " << blockResult.transpose();
}

TEST(addProduct, RowVector_x_BlockDiagonal)
{
    Matrix denseA{
        { 1, 2, 0,  0, 0, 0},
        {-1, 2, 0,  0, 0, 0},
        { 0, 0, 3, -5, 9, 0},
        { 0, 0, 2, -2, 2, 0},
        { 0, 0, 2,  1, 1, 0},
        { 0, 0, 0,  0, 0, 6}
    };

    const RowVector x{
        {1, 2, 3, 4, 5, 6}
    };

    RowVector denseResult{
        {1, -2, 3, -4, 5, -6}
    };
    RowVector blockResult  = denseResult;
    denseResult           += 2.0 * x * denseA;

    BlockDiagonalMatrix A(denseA, std::vector<int>{2, 3, 1});
    addProduct(2.0, x, A, blockResult);

    EXPECT_TRUE(denseResult.isApprox(blockResult))
        << "denseResult = " << denseResult << "\nblockResult = " << blockResult;

    denseResult = 2.0 * x * denseA;
    blockResult = product(2.0, x, A);

    EXPECT_TRUE(denseResult.isApprox(blockResult))
        << "denseResult = " << denseResult << "\nblockResult = " << blockResult;
}
