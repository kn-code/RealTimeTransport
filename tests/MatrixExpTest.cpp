//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#define REAL_TIME_TRANSPORT_DEBUG

#include <gtest/gtest.h>

#include <RealTimeTransport/BlockMatrices/MatrixExp.h>
#include <RealTimeTransport/Utility.h>

using namespace SciCore;
using namespace RealTimeTransport;

TEST(MatrixExp, RandomMatrix)
{
    Matrix A = Matrix::Random(16, 16);

    MatrixExp expA(A);

    RealVector tValues = RealVector::LinSpaced(500, 0, 100);
    for (Real t : tValues)
    {
        Matrix exp1         = expA(t);
        Matrix exp1Expected = RealTimeTransport::exp(t * A);

        EXPECT_TRUE(exp1.isApprox(exp1Expected));
    }
}

TEST(MatrixExp, NonSquareError)
{
    Matrix A = Matrix::Random(16, 14);
    MatrixExp expA;

    EXPECT_ANY_THROW(expA.initialize(A));
}

TEST(MatrixExp, NonDiagonalizable)
{
    Matrix A{
        {3, 1},
        {0, 3}
    };
    MatrixExp expA(A);
    Matrix expResult{
        {std::exp(6), 2 * std::exp(6)},
        {         0.,     std::exp(6)}
    };
    EXPECT_TRUE(expA(2.0).isApprox(expResult)) << "expA(2.0)=\n" << expA(2.0) << "\nexpResult=\n" << expResult;
}

TEST(BlockDiagonalMatrixExp, BasicTest)
{
    Matrix A{
        { 1, 2, 0,  0, 0, 0},
        {-1, 2, 0,  0, 0, 0},
        { 0, 0, 3, -5, 9, 0},
        { 0, 0, 2, -2, 2, 0},
        { 0, 0, 2,  1, 1, 0},
        { 0, 0, 0,  0, 0, 6}
    };

    std::vector<int> blockDims{2, 3, 1};

    BlockDiagonalMatrix Ablock(A, blockDims);
    BlockDiagonalMatrixExp expA(Ablock);

    RealVector tValues = RealVector::LinSpaced(500, 0, 100);
    for (Real t : tValues)
    {
        Matrix exp1         = expA(t).toDense();
        Matrix exp1Expected = RealTimeTransport::exp(t * A);

        EXPECT_TRUE(exp1.isApprox(exp1Expected));
    }
}

TEST(BlockDiagonalMatrixExp, expm1)
{
    Matrix A{
        { 1, 2, 0,  0, 0, 0},
        {-1, 2, 0,  0, 0, 0},
        { 0, 0, 3, -5, 9, 0},
        { 0, 0, 2, -2, 2, 0},
        { 0, 0, 2,  1, 1, 0},
        { 0, 0, 0,  0, 0, 6}
    };

    std::vector<int> blockDims{2, 3, 1};

    BlockDiagonalMatrix Ablock(A, blockDims);
    BlockDiagonalMatrixExp expA(Ablock);

    RealVector tValues = RealVector::LinSpaced(500, 0, 100);
    for (Real t : tValues)
    {
        Matrix exp1         = expA.expm1(t).toDense();
        Matrix exp1Expected = expm1(t * A);

        EXPECT_TRUE(exp1.isApprox(exp1Expected));
    }
}
