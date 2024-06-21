//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

///
/// \file   BlockDiagonalMatrix.h
///
/// \brief  Contains a class representing block diagonal matrices.
///

#ifndef REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_DIAGONAL_H
#define REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_DIAGONAL_H

#include <vector>

#include <SciCore/Definitions.h>

#include "../Error.h"
#include "../RealTimeTransport_export.h"
#include "BlockHelper.h"

namespace RealTimeTransport
{

///
/// @brief Represents a block diagonal matrix.
///
/// This class represents a complex valued block diagonal matrix.
///
class REALTIMETRANSPORT_EXPORT BlockDiagonalMatrix
{
  public:
    /// @brief Type representing the matrix elements.
    using Scalar = SciCore::Complex;

    /// @brief Type representing a matrix block.
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    ///
    /// @brief Constructs an empty block diagonal matrix.
    ///
    BlockDiagonalMatrix() noexcept;

    ///
    /// @brief Move constructor.
    ///
    BlockDiagonalMatrix(BlockDiagonalMatrix&& other) noexcept;

    ///
    /// @brief Copy constructor.
    ///
    BlockDiagonalMatrix(const BlockDiagonalMatrix& other);

    ///
    /// @brief Constructs a block diagonal matrix with given \a blocks.
    ///
    BlockDiagonalMatrix(std::vector<MatrixType>&& blocks) noexcept;

    ///
    /// @brief Constructs a block diagonal matrix from a dense \a matrix where the dimensions of the blocks are given by \a blockDimensions.
    ///
    template <typename DenseMatrixT>
    BlockDiagonalMatrix(const DenseMatrixT& matrix, const std::vector<int>& blockDimensions)
    {
        fromDense(matrix, blockDimensions);
    }

    ///
    /// @brief Move assignment operator.
    ///
    BlockDiagonalMatrix& operator=(BlockDiagonalMatrix&& other) noexcept;

    ///
    /// @brief Copy assignment operator.
    ///
    BlockDiagonalMatrix& operator=(const BlockDiagonalMatrix& other);

    ///
    /// @brief Equality comparison operator.
    ///
    bool operator==(const BlockDiagonalMatrix& other) const;

    ///
    /// @brief Inequality comparison operator.
    ///
    bool operator!=(const BlockDiagonalMatrix& other) const;

    ///
    /// @brief Returns a matrix with given \a blockDimensions that is zero everywhere.
    ///
    static BlockDiagonalMatrix Zero(const std::vector<int>& blockDimensions);

    ///
    /// @brief Returns a matrix with given \a blockDimensions equal to the identity matrix.
    ///
    static BlockDiagonalMatrix Identity(const std::vector<int>& blockDimensions);

    ///
    /// @brief Adds the block diagonal matrix \a rhs to the object.
    ///
    BlockDiagonalMatrix& operator+=(const BlockDiagonalMatrix& rhs);

    ///
    /// @brief Subtracts the block diagonal matrix \a rhs from the object.
    ///
    BlockDiagonalMatrix& operator-=(const BlockDiagonalMatrix& rhs);

    ///
    /// @brief Multiplies the block diagonal matrix \a rhs from the right to object.
    ///
    BlockDiagonalMatrix& operator*=(const BlockDiagonalMatrix& rhs);

    ///
    /// @brief Multiplies the object by the a \a scalar.
    ///
    BlockDiagonalMatrix& operator*=(SciCore::Real scalar);

    ///
    /// @brief Multiplies the object by the a \a scalar.
    ///
    BlockDiagonalMatrix& operator*=(SciCore::Complex scalar);

    ///
    /// @brief Multiplies the number of blocks.
    ///
    int numBlocks() const noexcept;

    ///
    /// @brief Returns the total number of rows.
    ///
    int totalRows() const noexcept;

    ///
    /// @brief Returns the total number of columns.
    ///
    int totalCols() const noexcept;

    ///
    /// @brief Returns a vector containing the dimensions of each block.
    ///
    std::vector<int> blockDimensions() const;

    ///
    /// @brief Returns the matrix block with index \a i.
    ///
    MatrixType& operator()(int i);

    ///
    /// @brief Returns the matrix block with index \a i.
    ///
    const MatrixType& operator()(int i) const;

    ///
    /// @brief Creates a new block diagonal matrix from given \a blocks.
    ///
    void fromBlocks(std::vector<MatrixType>&& newBlocks);

    ///
    /// @brief Constructs a block diagonal matrix from a dense \a matrix where the dimensions of the blocks are given by \a blockDimensions.
    ///
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

    ///
    /// @brief Sets the matrix to zero with block dimensions \a blockDimensions.
    ///
    void setZero(const std::vector<int>& blockDimensions);

    ///
    /// @brief Returns a dense matrix representation.
    ///
    MatrixType toDense() const;

    ///
    /// @brief Returns the matrix element at a given \a row and \a column.
    ///
    const Scalar& element(int row, int column) const;

    ///
    /// @brief Returns the matrix element at a given \a row and \a column.
    ///
    Scalar& element(int row, int colum);

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
