//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#define REAL_TIME_TRANSPORT_DEBUG

#include <gtest/gtest.h>

#include <RealTimeTransport/BlockMatrices/MatrixOperations.h>

using namespace SciCore;
using namespace RealTimeTransport;

TEST(addProduct, BlockMatrix_BlockMatrix_toDiagonal)
{
    Matrix A{
        {0,  0, 2, -1},
        {0,  0, 3,  4},
        {1, -2, 0,  0},
        {8,  1, 0,  0}
    };
    Matrix B{
        { 0,  0, 1, -1},
        { 0,  0, 4,  5},
        {-1, -2, 0,  0},
        {-6,  1, 0,  0}
    };

    Matrix denseResult{
        {1, 2, 0, 0},
        {3, 4, 0, 0},
        {0, 0, 5, 6},
        {0, 0, 7, 8}
    };

    std::vector<int> blockDims = {2, 2};
    BlockDiagonalMatrix blockResult(denseResult, blockDims);

    denseResult += 2.0 * A * B;

    BlockMatrix<Complex> blockA(A, blockDims);
    BlockMatrix<Complex> blockB(B, blockDims);

    addProduct(2.0, blockA, blockB, blockResult);

    EXPECT_TRUE(denseResult.isApprox(blockResult.toDense()));
}

TEST(addProduct, BlockMatrix_BlockMatrix_toDiagonal_2)
{
    Matrix A{
        { 1, 4, 0, 0, 0, 0},
        {-1, 2, 0, 0, 0, 0},
        { 0, 0, 7, 6, 1, 0},
        { 0, 0, 5, 5, 2, 0},
        { 0, 0, 2, 4, 1, 0},
        { 0, 0, 0, 0, 0, 2}
    };
    Matrix B = A;

    Matrix denseResult         = A;
    std::vector<int> blockDims = {2, 3, 1};
    BlockDiagonalMatrix blockResult(denseResult, blockDims);

    denseResult += 2.0 * A * B;

    BlockMatrix<Complex> blockA(A, blockDims);
    BlockMatrix<Complex> blockB(B, blockDims);

    addProduct(2.0, blockA, blockB, blockResult);

    EXPECT_TRUE(denseResult.isApprox(blockResult.toDense()));
}

TEST(addProduct, MatrixElement_BlockMatrix_BlockMatrix_BlockMatrix)
{
    Matrix A{
        {0,  0, 2, -1},
        {0,  0, 3,  4},
        {1, -2, 0,  0},
        {8,  1, 0,  0}
    };
    Matrix B{
        { 0,  0, 1, -1},
        { 0,  0, 4,  5},
        {-1, -2, 0,  0},
        {-6,  1, 0,  0}
    };
    Matrix C{
        {3, -4, 0,  0},
        {0,  0, 4,  5},
        {7,  8, 0,  0},
        {0,  0, 9, 10}
    };

    Matrix denseResult  = C;
    denseResult        += 2.0 * A * B * C;

    std::vector<int> blockDims = {2, 2};
    BlockMatrix<Complex> blockA(A, blockDims);
    BlockMatrix<Complex> blockB(B, blockDims);
    BlockMatrix<Complex> blockC(C, blockDims);

    Matrix blockResult{
        {3, -4},
        {0,  0}
    };
    addProduct(0, 0, 2.0, blockA, blockB, blockC, blockResult);
    EXPECT_TRUE(denseResult.block(0, 0, 2, 2).isApprox(blockResult));

    blockResult = Matrix{
        {0, 0},
        {4, 5}
    };
    addProduct(0, 1, 2.0, blockA, blockB, blockC, blockResult);
    EXPECT_TRUE(denseResult.block(0, 2, 2, 2).isApprox(blockResult));

    blockResult = Matrix{
        {7, 8},
        {0, 0}
    };
    addProduct(1, 0, 2.0, blockA, blockB, blockC, blockResult);
    EXPECT_TRUE(denseResult.block(2, 0, 2, 2).isApprox(blockResult));

    blockResult = Matrix{
        {0,  0},
        {9, 10}
    };
    addProduct(1, 1, 2.0, blockA, blockB, blockC, blockResult);
    EXPECT_TRUE(denseResult.block(2, 2, 2, 2).isApprox(blockResult));
}

TEST(addProduct_col, BlockMatrix_BlockDiagonal_BlockMatrix)
{
    Matrix A1{
        {0, 0, 0,  0, 0, 1},
        {0, 0, 0,  0, 0, 6},
        {0, 0, 1, -1, 1, 0},
        {0, 0, 2,  2, 2, 0},
        {0, 0, 3, -3, 3, 0},
        {0, 0, 0,  0, 0, 0}
    };
    Matrix D{
        { 1, 4, 0, 0, 0, 0},
        {-1, 2, 0, 0, 0, 0},
        { 0, 0, 7, 6, 1, 0},
        { 0, 0, 5, 5, 2, 0},
        { 0, 0, 2, 4, 1, 0},
        { 0, 0, 0, 0, 0, 2}
    };
    Matrix A2 = A1.transpose();

    Matrix result  = D;
    result        += 2.0 * A1 * D * A2;

    std::vector<int> blockDims = {2, 3, 1};
    BlockMatrix<Complex> blockA1(A1, blockDims);
    BlockDiagonalMatrix blockD(D, blockDims);
    BlockMatrix<Complex> blockA2(A2, blockDims);

    BlockVector<Complex> resultBlock;
    resultBlock.emplace(
        0, Matrix{
               { 1, 4},
               {-1, 2}
    });

    addProduct_col(0, 2.0, blockA1, blockD, blockA2, resultBlock);

    ASSERT_TRUE(resultBlock.contains(0)) << "result=\n" << result;
    ASSERT_TRUE(resultBlock.contains(1) == false) << "result=\n" << result;
    ASSERT_TRUE(resultBlock.contains(2) == false) << "result=\n" << result;

    EXPECT_TRUE(result.block(0, 0, 2, 2).isApprox(resultBlock.find(0)->second));
}

TEST(addProduct_col_unsafe, BlockMatrix_BlockMatrix_BlockMatrix)
{
    Matrix A{
        {0,  0, 2, -1},
        {0,  0, 3,  4},
        {1, -2, 0,  0},
        {8,  1, 0,  0}
    };
    Matrix B{
        { 0,  0, 1, -1},
        { 0,  0, 4,  5},
        {-1, -2, 0,  0},
        {-6,  1, 0,  0}
    };
    Matrix C{
        {3, -4, 0,  0},
        {0,  0, 4,  5},
        {7,  8, 0,  0},
        {0,  0, 9, 10}
    };

    Matrix denseResult  = C;
    denseResult        += 2.0 * A * B * C;

    std::vector<int> blockDims = {2, 2};
    BlockMatrix<Complex> blockA(A, blockDims);
    BlockMatrix<Complex> blockB(B, blockDims);
    BlockMatrix<Complex> blockC(C, blockDims);

    {
        BlockVector<Complex> result;
        addProduct_col_unsafe(0, 2.0, blockA, blockB, blockC, result);
        EXPECT_EQ(result.size(), 0);
    }

    {
        BlockVector<Complex> result;
        result.emplace(
            0, Matrix{
                   {3, -4},
                   {0,  0}
        });
        addProduct_col_unsafe(0, 2.0, blockA, blockB, blockC, result);
        EXPECT_EQ(result.size(), 1);
        ASSERT_EQ(result.contains(0), true);
        EXPECT_TRUE(denseResult.block(0, 0, 2, 2).isApprox(result.find(0)->second));
    }

    {
        BlockVector<Complex> result;
        result.emplace(
            1, Matrix{
                   {7, 8},
                   {0, 0}
        });
        addProduct_col_unsafe(0, 2.0, blockA, blockB, blockC, result);
        EXPECT_EQ(result.size(), 1);
        ASSERT_EQ(result.contains(1), true);
        EXPECT_TRUE(denseResult.block(2, 0, 2, 2).isApprox(result.find(1)->second));
    }

    {
        BlockVector<Complex> result;
        result.emplace(
            0, Matrix{
                   {0, 0},
                   {4, 5}
        });
        result.emplace(
            1, Matrix{
                   {0,  0},
                   {9, 10}
        });
        addProduct_col_unsafe(1, 2.0, blockA, blockB, blockC, result);
        ASSERT_EQ(result.size(), 2);
        ASSERT_EQ(result.contains(0), true);
        ASSERT_EQ(result.contains(1), true);
        EXPECT_TRUE(denseResult.block(0, 2, 2, 2).isApprox(result.find(0)->second));
        EXPECT_TRUE(denseResult.block(2, 2, 2, 2).isApprox(result.find(1)->second));
    }
}

TEST(product, BlockMatrix_BlockMatrix_BlockMatrix_1)
{
    const Matrix A = Matrix::Random(64, 64);
    const Matrix B = Matrix::Random(64, 64);
    const Matrix C = Matrix::Random(64, 64);

    std::vector<int> blockDims = {16, 16, 8, 16, 8};
    const BlockMatrix<Complex> blockA(A, blockDims);
    const BlockMatrix<Complex> blockB(B, blockDims);
    const BlockMatrix<Complex> blockC(C, blockDims);

    EXPECT_EQ(blockA.toDense(), A);
    EXPECT_EQ(blockB.toDense(), B);
    EXPECT_EQ(blockC.toDense(), C);

    Matrix denseResult               = 2.0 * A * B * C;
    BlockMatrix<Complex> blockResult = product(2.0, blockA, blockB, blockC);
    EXPECT_TRUE(denseResult.isApprox(blockResult.toDense()));
}

TEST(product, BlockMatrix_BlockMatrix_BlockMatrix_2)
{
    const Matrix A{
        { 1, 4, 0, 0, 0, 0},
        {-1, 2, 0, 0, 0, 0},
        { 0, 0, 7, 6, 1, 0},
        { 0, 0, 5, 5, 2, 0},
        { 0, 0, 2, 4, 1, 0},
        { 1, 3, 0, 0, 0, 2}
    };
    const Matrix B{
        { 0, 0, 1, -1, 1, 0},
        { 0, 0, 4,  5, 5, 0},
        { 0, 0, 1,  1, 1, 0},
        {-6, 1, 0,  0, 0, 0},
        { 6, 1, 0,  0, 0, 0},
        { 4, 5, 0,  0, 0, 1}
    };
    const Matrix C = A;

    std::vector<int> blockDims = {2, 3, 1};
    const BlockMatrix<Complex> blockA(A, blockDims);
    const BlockMatrix<Complex> blockB(B, blockDims);
    const BlockMatrix<Complex> blockC(C, blockDims);

    EXPECT_EQ(blockA.toDense(), A);
    EXPECT_EQ(blockB.toDense(), B);
    EXPECT_EQ(blockC.toDense(), C);

    Matrix denseResult               = 2.0 * A * B * C;
    BlockMatrix<Complex> blockResult = product(2.0, blockA, blockB, blockC);
    EXPECT_TRUE(denseResult.isApprox(blockResult.toDense())) << "denseResult =\n"
                                                             << denseResult << "\nblockResult =\n"
                                                             << blockResult.toDense();

    denseResult = 2.0 * A * C * B;
    blockResult = product(2.0, blockA, blockC, blockB);
    EXPECT_TRUE(denseResult.isApprox(blockResult.toDense())) << "denseResult =\n"
                                                             << denseResult << "\nblockResult =\n"
                                                             << blockResult.toDense();
}

TEST(addProduct, BlockMatrix_BlockDiagonal_BlockMatrix)
{
    std::vector<int> blockDims{32, 64, 28, 56, 1, 44, 31};
    std::vector<Matrix> blocks;
    for (int dim : blockDims)
    {
        blocks.push_back(Matrix::Random(dim, dim));
    }

    BlockDiagonalMatrix D(std::vector<Matrix>{blocks});

    int totalDim   = 256;
    Matrix Bdense  = Matrix::Zero(totalDim, totalDim);
    int blockStart = 0;
    for (int i = 0; i < D.numBlocks(); ++i)
    {
        Bdense.block(blockStart, blockStart, blockDims[i], blockDims[i]) = blocks[i];

        blockStart += blockDims[i];
    }

    Matrix A1dense = Matrix::Random(totalDim, totalDim);
    Matrix A2dense = Matrix::Random(totalDim, totalDim);

    BlockMatrix<Complex> A1(A1dense, blockDims);
    BlockMatrix<Complex> A2(A2dense, blockDims);

    Matrix denseResult  = Matrix::Constant(totalDim, totalDim, 1.0);
    denseResult        += 2.0 * A1dense * Bdense * A2dense;

    int numBlocks = (int)blockDims.size();
    int startRow  = 0;
    for (int i = 0; i < numBlocks; ++i)
    {
        int startCol = 0;
        for (int j = 0; j < numBlocks; ++j)
        {
            Matrix denseResult_ij = denseResult.block(startRow, startCol, blockDims[i], blockDims[j]);

            Matrix blockResult_ij = Matrix::Constant(blockDims[i], blockDims[j], 1.0);
            addProduct(i, j, 2.0, A1, D, A2, blockResult_ij);

            EXPECT_TRUE(denseResult_ij.isApprox(blockResult_ij));

            startCol += blockDims[j];
        }
        // Update startRow for the next row of blocks
        startRow += blockDims[i];
    }
}

TEST(product, BlockDiagonal_BlockDiagonal_BlockDiagonal)
{
    Matrix A{
        { 1, 4, 0, 0, 0, 0},
        {-1, 2, 0, 0, 0, 0},
        { 0, 0, 7, 6, 1, 0},
        { 0, 0, 5, 5, 2, 0},
        { 0, 0, 2, 4, 1, 0},
        { 0, 0, 0, 0, 0, 2}
    };

    std::vector<int> blockDims = {2, 3, 1};
    BlockDiagonalMatrix Ablock(A, blockDims);

    Matrix denseResult              = 2.0 * A * A * A;
    BlockDiagonalMatrix blockResult = product(2.0, Ablock, Ablock, Ablock);

    EXPECT_TRUE(denseResult.isApprox(blockResult.toDense()));
}

TEST(addProduct, RowVector_BlockDiagonal_BlockMatrix)
{
    const Matrix A{
        { 1, 4, 0, 0, 0, 4},
        {-1, 2, 0, 0, 0, 0},
        { 0, 0, 7, 6, 1, 0},
        { 0, 0, 5, 5, 2, 0},
        { 0, 0, 2, 4, 1, 0},
        { 1, 3, 0, 0, 0, 2}
    };

    const Matrix D{
        {9, 7, 0, 0, 0, 0},
        {8, 1, 0, 0, 0, 0},
        {0, 0, 1, 6, 2, 0},
        {0, 0, 3, 3, 0, 0},
        {0, 0, 2, 0, 1, 0},
        {0, 0, 0, 0, 0, 7}
    };

    const RowVector x{
        {1, 2, 3, 4, 5, 6}
    };

    std::vector<int> blockDims = {2, 3, 1};
    const BlockMatrix<Complex> blockA(A, blockDims);
    const BlockDiagonalMatrix blockD(D, blockDims);

    std::vector<int> blockStartIndices(blockDims.size(), 0);
    for (size_t i = 1; i < blockDims.size(); ++i)
    {
        blockStartIndices[i] = blockStartIndices[i - 1] + blockDims[i - 1];
    }

    RowVector denseResult  = x;
    denseResult           += 2.0 * x * D * A;

    RowVector blockResult = x;
    addProduct(2.0, x, blockD, blockA, blockResult, blockStartIndices);

    EXPECT_TRUE(denseResult.isApprox(blockResult));
}

TEST(addProduct, MatrixElement_BlockMatrix_BlockMatrix_BlockVector)
{
    Matrix A{
        {0,  0, 2, -1},
        {0,  0, 3,  4},
        {1, -2, 0,  0},
        {8,  1, 0,  0}
    };
    Matrix B{
        { 0,  0, 1, -1},
        { 0,  0, 4,  5},
        {-1, -2, 0,  0},
        {-6,  1, 0,  0}
    };
    Matrix C{
        {3, -4},
        {0,  0},
        {7,  8},
        {0,  0}
    };

    Matrix denseResult  = C;
    denseResult        += 2.0 * A * B * C;

    std::vector<int> blockDims = {2, 2};
    BlockMatrix<Complex> blockA(A, blockDims);
    BlockMatrix<Complex> blockB(B, blockDims);
    BlockVector<Complex> x;
    x.emplace(
        0, Matrix{
               {3, -4},
               {0,  0},
    });
    x.emplace(
        1, Matrix{
               {7, 8},
               {0, 0},
    });

    Matrix blockResult{
        {3, -4},
        {0,  0}
    };
    addProduct(0, 2.0, blockA, blockB, x, blockResult);
    EXPECT_TRUE(denseResult.block(0, 0, 2, 2).isApprox(blockResult));

    blockResult = Matrix{
        {7, 8},
        {0, 0}
    };
    addProduct(1, 2.0, blockA, blockB, x, blockResult);
    EXPECT_TRUE(denseResult.block(2, 0, 2, 2).isApprox(blockResult));
}

/*TEST(product, BlockDiagonal_BlockMatrix_BlockDiagonal)
{
    std::vector<int> blockDims{32, 64, 28, 56, 1, 44, 31};
    std::vector<Matrix> blocks1, blocks2;
    for (int dim : blockDims)
    {
        blocks1.push_back(Matrix::Random(dim, dim));
        blocks2.push_back(Matrix::Random(dim, dim));
    }

    BlockDiagonalMatrix D1(std::vector<Matrix>{blocks1});
    BlockDiagonalMatrix D2(std::vector<Matrix>{blocks2});

    BlockMatrix<Complex>::UnorderedElementMap elements{
        {{0, 1}, Matrix::Random(blockDims[0], blockDims[1])},
        {{3, 2}, Matrix::Random(blockDims[3], blockDims[2])},
        {{4, 6}, Matrix::Random(blockDims[4], blockDims[6])}
    };

    BlockMatrix<Complex> A(std::move(elements), blockDims);

    Matrix denseResult               = 2.0 * D1.toDense() * A.toDense() * D2.toDense();
    BlockMatrix<Complex> blockResult = product(2.0, D1, A, D2);

    EXPECT_TRUE(denseResult.isApprox(blockResult.toDense()));
}*/

/*TEST(product, BlockMatrix_BlockMatrix_1)
{
    const Matrix A{
        {2, -1, 0, 0},
        {3,  4, 0, 0},
        {1, -2, 0, 0},
        {8,  1, 0, 0}
    };
    const Matrix B{
        { 0,  0, 1, -1},
        { 0,  0, 4,  5},
        {-1, -2, 0,  0},
        {-6,  1, 0,  0}
    };

    std::vector<int> blockDims = {2, 2};
    const BlockMatrix<Complex> blockA(A, blockDims);
    const BlockMatrix<Complex> blockB(B, blockDims);

    Matrix denseResult               = 2.0 * A * B;
    BlockMatrix<Complex> blockResult = product(2.0, blockA, blockB);
    EXPECT_TRUE(denseResult.isApprox(blockResult.toDense()));

    denseResult = 2.0 * B * A;
    blockResult = product(2.0, blockB, blockA);
    EXPECT_TRUE(denseResult.isApprox(blockResult.toDense())) << "denseResult =\n"
                                                             << denseResult << "\nblockResult =\n"
                                                             << blockResult.toDense();
}

TEST(product, BlockMatrix_BlockMatrix_2)
{
    const Matrix A{
        { 1, 4, 0, 0, 0, 0},
        {-1, 2, 0, 0, 0, 0},
        { 0, 0, 7, 6, 1, 0},
        { 0, 0, 5, 5, 2, 0},
        { 0, 0, 2, 4, 1, 0},
        { 1, 3, 0, 0, 0, 2}
    };
    const Matrix B{
        { 0, 0, 1, -1, 1, 0},
        { 0, 0, 4,  5, 5, 0},
        { 0, 0, 1,  1, 1, 0},
        {-6, 1, 0,  0, 0, 0},
        { 6, 1, 0,  0, 0, 0},
        { 4, 5, 0,  0, 0, 1}
    };

    std::vector<int> blockDims = {2, 3, 1};
    const BlockMatrix<Complex> blockA(A, blockDims);
    const BlockMatrix<Complex> blockB(B, blockDims);

    EXPECT_EQ(blockA.toDense(), A);
    EXPECT_EQ(blockB.toDense(), B);

    Matrix denseResult               = 2.0 * A * B;
    BlockMatrix<Complex> blockResult = product(2.0, blockA, blockB);
    EXPECT_TRUE(denseResult.isApprox(blockResult.toDense())) << "denseResult =\n"
                                                             << denseResult << "\nblockResult =\n"
                                                             << blockResult.toDense();

    denseResult = 2.0 * B * A;
    blockResult = product(2.0, blockB, blockA);
    EXPECT_TRUE(denseResult.isApprox(blockResult.toDense())) << "denseResult =\n"
                                                             << denseResult << "\nblockResult =\n"
                                                             << blockResult.toDense();
}

TEST(product, BlockMatrix_BlockMatrix_3)
{
    const Matrix A = Matrix::Random(128, 128);
    const Matrix B = Matrix::Random(128, 128);

    std::vector<int> blockDims = {32, 16, 64, 16};
    const BlockMatrix<Complex> blockA(A, blockDims);
    const BlockMatrix<Complex> blockB(B, blockDims);

    EXPECT_EQ(blockA.toDense(), A);
    EXPECT_EQ(blockB.toDense(), B);

    Matrix denseResult               = 2.0 * A * B;
    BlockMatrix<Complex> blockResult = product(2.0, blockA, blockB);
    EXPECT_TRUE(denseResult.isApprox(blockResult.toDense()));

    denseResult = 2.0 * B * A;
    blockResult = product(2.0, blockB, blockA);
    EXPECT_TRUE(denseResult.isApprox(blockResult.toDense()));
}

TEST(product, BlockMatrix_BlockMatrix_4)
{
    // A^2 == 0
    const Matrix A{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, -1,  0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  1,  0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  0, -1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  0,  1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 1, -1,  0,  0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  0,  0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0,  0,  0,  0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  0,  0},
        {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  0,  0},
        {0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  0,  0},
        {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  0, 0,  0,  0,  0},
        {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  0, 0,  0,  0,  0},
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  0, 0,  0,  0,  0},
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  0, 0,  0,  0,  0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  0,  0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  0,  0}
    };

    std::vector<int> blockDims = {4, 1, 1, 1, 1, 2, 2, 2, 2};
    const BlockMatrix<Complex> blockA(A, blockDims);

    BlockMatrix<Complex> blockResult = product(2.0, blockA, blockA);
    EXPECT_EQ(blockResult.size(), 0);
    EXPECT_EQ(blockResult.toDense(), Matrix::Zero(16, 16));
}*/