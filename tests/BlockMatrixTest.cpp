//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#define REAL_TIME_TRANSPORT_DEBUG

#include <gtest/gtest.h>

#include <RealTimeTransport/BlockMatrices/BlockMatrix.h>

using namespace SciCore;
using namespace RealTimeTransport;

TEST(BlockMatrix, fromUnorderedMap)
{
    Matrix Adense{
        {0, 0, 1, 2, 0},
        {0, 0, 3, 4, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {5, 6, 0, 0, 0}
    };

    std::vector<int> blockDims = {2, 2, 1};
    BlockMatrix::UnorderedElementMap elements{
        {{0, 1}, Matrix{{1, 2}, {3, 4}}},
        {{2, 0},         Matrix{{5, 6}}}
    };

    // Create a BlockMatrix from triplets
    BlockMatrix A(std::move(elements), blockDims);
    EXPECT_EQ(A.size(), 2);

    // Check if the original and converted matrices are equal
    EXPECT_EQ(Adense, A.toDense());
}

TEST(BlockMatrix, ConvertToAndFromDense)
{
    Matrix originalMatrix = Matrix::Random(12, 12);

    std::vector<int> blockDims = {2, 4, 1, 3, 2};

    // Create a BlockMatrix from the dense matrix
    BlockMatrix A(originalMatrix, blockDims);
    EXPECT_EQ(A.size(), blockDims.size() * blockDims.size());

    // Convert back to a dense matrix
    Matrix convertedMatrix = A.toDense();

    // Check if the original and converted matrices are equal
    EXPECT_EQ(originalMatrix, convertedMatrix);
}

TEST(BlockMatrix, operatorAddAssign)
{
    Matrix Adense{
        {0, 0, 1, 2, 0},
        {0, 0, 3, 4, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {5, 6, 0, 0, 0}
    };

    Matrix Bdense = Matrix::Random(5, 5);

    Matrix expected = Adense + Bdense;

    std::vector<int> blockDims = {2, 2, 1};
    BlockMatrix A(Adense, blockDims);
    BlockMatrix B(Bdense, blockDims);
    A += B;

    EXPECT_EQ(expected, A.toDense());
}

TEST(addProduct, BlockMatrix_Vector)
{
    const Matrix A{
        { 1, 4, 0, 0, 0, 0},
        {-1, 2, 0, 0, 0, 0},
        { 0, 0, 7, 6, 1, 0},
        { 0, 0, 5, 5, 2, 0},
        { 0, 0, 2, 4, 1, 0},
        { 1, 3, 0, 0, 0, 2}
    };

    std::vector<int> blockDims = {2, 3, 1};
    const BlockMatrix blockA(A, blockDims);

    std::vector<int> blockStartIndices(blockDims.size(), 0);
    for (size_t i = 1; i < blockDims.size(); ++i)
    {
        blockStartIndices[i] = blockStartIndices[i - 1] + blockDims[i - 1];
    }

    Vector x{
        {1, 2, 3, 4, 5, 6}
    };

    Vector resultDense  = x;
    resultDense        += 2 * A * x;

    Vector resultBlock = x;
    addProduct(2.0, blockA, x, resultBlock, blockStartIndices);

    EXPECT_TRUE(resultDense.isApprox(resultBlock));
}

TEST(addProduct, RowVector_BlockMatrix)
{
    const Matrix A{
        { 1, 4, 0, 0, 0, 0},
        {-1, 2, 0, 0, 0, 0},
        { 0, 0, 7, 6, 1, 0},
        { 0, 0, 5, 5, 2, 0},
        { 0, 0, 2, 4, 1, 0},
        { 1, 3, 0, 0, 0, 2}
    };

    std::vector<int> blockDims = {2, 3, 1};
    const BlockMatrix blockA(A, blockDims);

    std::vector<int> blockStartIndices(blockDims.size(), 0);
    for (size_t i = 1; i < blockDims.size(); ++i)
    {
        blockStartIndices[i] = blockStartIndices[i - 1] + blockDims[i - 1];
    }

    RowVector x{
        {1, 2, 3, 4, 5, 6}
    };

    RowVector resultDense  = x;
    resultDense           += 2 * x * A;

    RowVector resultBlock = x;
    addProduct(2.0, x, blockA, resultBlock, blockStartIndices);

    EXPECT_TRUE(resultDense.isApprox(resultBlock));
}