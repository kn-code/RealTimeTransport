//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_DIAGONAL_H
#define REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_DIAGONAL_H

#include <vector>

#include <SciCore/Definitions.h>

#include "../Error.h"
#include "../RealTimeTransport_export.h"
#include "BlockHelper.h"

namespace RealTimeTransport
{

class REALTIMETRANSPORT_EXPORT BlockDiagonalMatrix
{
  public:
    using Scalar     = SciCore::Complex;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    BlockDiagonalMatrix() noexcept;
    BlockDiagonalMatrix(std::vector<MatrixType>&& newBlocks) noexcept;

    template <typename DenseMatrixT>
    BlockDiagonalMatrix(const DenseMatrixT& matrix, const std::vector<int>& blockDimensions)
    {
        fromDense(matrix, blockDimensions);
    }

    BlockDiagonalMatrix(BlockDiagonalMatrix&& other) noexcept;
    BlockDiagonalMatrix(const BlockDiagonalMatrix& other);

    bool operator==(const BlockDiagonalMatrix& other) const;
    bool operator!=(const BlockDiagonalMatrix& other) const;

    static BlockDiagonalMatrix Zero(const std::vector<int>& blockDimensions);
    static BlockDiagonalMatrix Identity(const std::vector<int>& blockDimensions);

    BlockDiagonalMatrix& operator=(BlockDiagonalMatrix&& other) noexcept;
    BlockDiagonalMatrix& operator=(const BlockDiagonalMatrix& other);

    BlockDiagonalMatrix& operator+=(const BlockDiagonalMatrix& rhs);
    BlockDiagonalMatrix& operator-=(const BlockDiagonalMatrix& rhs);

    BlockDiagonalMatrix& operator*=(const BlockDiagonalMatrix& rhs);
    BlockDiagonalMatrix& operator*=(SciCore::Real scalar);
    BlockDiagonalMatrix& operator*=(SciCore::Complex scalar);

    int numBlocks() const noexcept;
    int totalRows() const noexcept;
    int totalCols() const noexcept;

    std::vector<int> blockDimensions() const;

    MatrixType& operator()(int i);
    const MatrixType& operator()(int i) const;

    void fromBlocks(std::vector<MatrixType>&& newBlocks);

    template <typename DenseMatrixT>
    void fromDense(const DenseMatrixT& matrix, const std::vector<int>& blockDimensions)
    {
#ifdef REAL_TIME_TRANSPORT_DEBUG
        if (isBlockDiagonal(matrix, blockDimensions) == false)
        {
            throw Error("Matrix is not block diagonal as required");
        }
#endif

        size_t numBlocks = blockDimensions.size();
        _blocks.resize(numBlocks);

        // Partition the matrix into blocks and store them
        for (int i = 0; i < static_cast<int>(numBlocks); ++i)
        {
            _blocks[i] = extractDenseBlock(matrix, i, i, blockDimensions);
        }
    }

    void setZero(const std::vector<int>& blockDimensions);

    MatrixType toDense() const;

    // Member function to return the element at a given row and column
    const Scalar& element(int row, int col) const;
    Scalar& element(int row, int col);

    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(_blocks);
    }

  private:
    std::vector<MatrixType> _blocks; // Store each block of the matrix
};

REALTIMETRANSPORT_EXPORT SciCore::Real maxNorm(const BlockDiagonalMatrix& A);

REALTIMETRANSPORT_EXPORT BlockDiagonalMatrix operator+(const BlockDiagonalMatrix& lhs, const BlockDiagonalMatrix& rhs);
REALTIMETRANSPORT_EXPORT BlockDiagonalMatrix operator-(const BlockDiagonalMatrix& lhs, const BlockDiagonalMatrix& rhs);
REALTIMETRANSPORT_EXPORT BlockDiagonalMatrix operator*(const BlockDiagonalMatrix& lhs, const BlockDiagonalMatrix& rhs);
REALTIMETRANSPORT_EXPORT BlockDiagonalMatrix operator*(const BlockDiagonalMatrix& lhs, SciCore::Real scalar);
REALTIMETRANSPORT_EXPORT BlockDiagonalMatrix operator*(const BlockDiagonalMatrix& lhs, SciCore::Complex scalar);
REALTIMETRANSPORT_EXPORT BlockDiagonalMatrix operator*(SciCore::Real scalar, const BlockDiagonalMatrix& rhs);
REALTIMETRANSPORT_EXPORT BlockDiagonalMatrix operator*(SciCore::Complex scalar, const BlockDiagonalMatrix& rhs);

// Computes result +=  α * A * x, where α is a scalar, A is a block diagonal matrix and x is a vector
template <typename ScalarT, typename T>
void addProduct(
    ScalarT alpha,
    const BlockDiagonalMatrix& A,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& result)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (A.totalCols() != x.size() || A.totalRows() != result.size())
    {
        throw Error("Vector size does not match the total size of the block diagonal matrix.");
    }
#endif // REAL_TIME_TRANSPORT_DEBUG

    int currentRow = 0;
    for (int i = 0; i < A.numBlocks(); ++i)
    {
        const auto& block                         = A(i);
        result.segment(currentRow, block.rows()) += alpha * block * x.segment(currentRow, block.rows());

        currentRow += block.rows();
    }
}

// Returns  α * A * x, where α is a scalar, A is a block diagonal matrix and x is a vector
template <typename ScalarT, typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> product(
    ScalarT alpha,
    const BlockDiagonalMatrix& A,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& x)
{
    using ReturnType = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    ReturnType returnValue = ReturnType::Zero(x.size());
    addProduct(alpha, A, x, returnValue);
    return returnValue;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> operator*(
    const BlockDiagonalMatrix& A,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& x)
{
    return product(1.0, A, x);
}

// Computes result +=  α * x * A, where α is a scalar, A is a block diagonal matrix and x is a rowvector
template <typename ScalarT, typename T>
void addProduct(
    ScalarT alpha,
    const Eigen::Matrix<T, 1, Eigen::Dynamic>& x,
    const BlockDiagonalMatrix& A,
    Eigen::Matrix<T, 1, Eigen::Dynamic>& result)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (A.totalRows() != x.size() || A.totalCols() != result.size())
    {
        throw Error("Vector size does not match the total size of the block diagonal matrix.");
    }
#endif // REAL_TIME_TRANSPORT_DEBUG

    int currentCol = 0;
    for (int i = 0; i < A.numBlocks(); ++i)
    {
        const auto& block                         = A(i);
        result.segment(currentCol, block.cols()) += alpha * x.segment(currentCol, block.cols()) * block;

        currentCol += block.cols();
    }
}

// Returns  α * x * A, where α is a scalar, A is a block diagonal matrix and x is a rowvector
template <typename ScalarT, typename T>
Eigen::Matrix<T, 1, Eigen::Dynamic> product(
    ScalarT alpha,
    const Eigen::Matrix<T, 1, Eigen::Dynamic>& x,
    const BlockDiagonalMatrix& A)
{
    using ReturnType = Eigen::Matrix<T, 1, Eigen::Dynamic>;

    ReturnType returnValue = ReturnType::Zero(x.size());
    addProduct(alpha, x, A, returnValue);
    return returnValue;
}

template <typename T>
Eigen::Matrix<T, 1, Eigen::Dynamic> operator*(
    const Eigen::Matrix<T, 1, Eigen::Dynamic>& x,
    const BlockDiagonalMatrix& A)
{

    return product(1.0, x, A);
}

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_DIAGONAL_H
