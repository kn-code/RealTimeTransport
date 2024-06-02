//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/BlockMatrices/MatrixExp.h"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <SciCore/BasicMath.h>

#include "RealTimeTransport/Utility.h"

namespace RealTimeTransport
{

MatrixExp::MatrixExp(const SciCore::Matrix& A)
{
    initialize(A);
}

void MatrixExp::initialize(const SciCore::Matrix& A)
{
    using namespace SciCore;

    _useEigendecomposition = true;

    int rows = A.rows();
    if (rows != A.cols())
    {
        throw Error("Matrix is not square");
    }

    Eigen::ComplexEigenSolver<Matrix> ces;
    ces.compute(A);

    const Matrix& rightEigenvectors = ces.eigenvectors();
    _eigenvalues                    = ces.eigenvalues();

    // Check if A is diagonalizable
    Real tol = 1e-12;
    Eigen::JacobiSVD<Matrix> svd(rightEigenvectors, Eigen::ComputeThinU | Eigen::ComputeThinV);
    int rank = (svd.singularValues().array() > tol).count();
    if (rank != rows)
    {
        _useEigendecomposition = false;
        _eigenvalues.resize(0);
        _projectors.resize(1);
        _projectors[0] = A;
        return;
    }

    Matrix leftEigenvectors = ces.eigenvectors().inverse();

    _projectors.resize(rows);
    for (int i = 0; i < rows; ++i)
    {
        _projectors[i] = rightEigenvectors.col(i) * leftEigenvectors.row(i);
    }
}

SciCore::Matrix MatrixExp::operator()(SciCore::Real t) const
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (_projectors.size() == 0)
    {
        throw Error("Object must be initialized before operator() is used");
    }
#endif

    if (_useEigendecomposition == true)
    {
        SciCore::Matrix returnValue = std::exp(t * _eigenvalues[0]) * _projectors[0];
        for (int i = 1; i < _eigenvalues.size(); ++i)
        {
            returnValue += std::exp(t * _eigenvalues[i]) * _projectors[i];
        }
        return returnValue;
    }
    else
    {
        return RealTimeTransport::exp(t * _projectors[0]);
    }
}

SciCore::Matrix MatrixExp::expm1(SciCore::Real t) const
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (_projectors.size() == 0)
    {
        throw Error("Object must be initialized before operator() is used");
    }
#endif

    if (_useEigendecomposition == true)
    {
        SciCore::Matrix returnValue = SciCore::expm1(t * _eigenvalues[0]) * _projectors[0];
        for (int i = 1; i < _eigenvalues.size(); ++i)
        {
            returnValue += SciCore::expm1(t * _eigenvalues[i]) * _projectors[i];
        }
        return returnValue;
    }
    else
    {
        return RealTimeTransport::expm1(t * _projectors[0]);
    }
}

BlockDiagonalMatrixExp::BlockDiagonalMatrixExp(const BlockDiagonalMatrix& A)
{
    initialize(A);
}

void BlockDiagonalMatrixExp::initialize(const BlockDiagonalMatrix& A)
{
    int numBlocks = A.numBlocks();
    _blocks.reserve(numBlocks);

    for (int i = 0; i < numBlocks; ++i)
    {
        _blocks.push_back(MatrixExp(A(i)));
    }
}

BlockDiagonalMatrix BlockDiagonalMatrixExp::operator()(SciCore::Real t) const
{
    using namespace SciCore;

    size_t numBlocks = _blocks.size();
    std::vector<Matrix> blocks(numBlocks);

    for (size_t i = 0; i < numBlocks; ++i)
    {
        blocks[i] = _blocks[i](t);
    }

    return BlockDiagonalMatrix(std::move(blocks));
}

BlockDiagonalMatrix BlockDiagonalMatrixExp::expm1(SciCore::Real t) const
{
    using namespace SciCore;

    size_t numBlocks = _blocks.size();
    std::vector<Matrix> blocks(numBlocks);

    for (size_t i = 0; i < numBlocks; ++i)
    {
        blocks[i] = _blocks[i].expm1(t);
    }

    return BlockDiagonalMatrix(std::move(blocks));
}

} // namespace RealTimeTransport
