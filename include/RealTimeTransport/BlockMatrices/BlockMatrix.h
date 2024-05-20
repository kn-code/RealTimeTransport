//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_MATRIX_H
#define REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_MATRIX_H

#include <set>
#include <vector>

#include "../extern/boost_unordered.hpp"

#include <SciCore/Definitions.h>

#include "../Error.h"
#include "../RealTimeTransport_export.h"
#include "BlockHelper.h"

namespace RealTimeTransport
{

// Represents a matrix separated into blocks, out of which most blocks are zero.
template <typename T>
class REALTIMETRANSPORT_EXPORT BlockMatrix
{
  public:
    using MatrixType          = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using Scalar              = T;
    using UnorderedElementMap = boost::unordered_flat_map<std::pair<int, int>, MatrixType>;

    BlockMatrix() noexcept
    {
    }

    BlockMatrix(const std::vector<int>& blockDimensions) : _blockDims(blockDimensions)
    {
        int numBlocks = this->numBlocks();
        _nonZeroRows.resize(numBlocks);
        _nonZeroCols.resize(numBlocks);
    }

    template <SciCore::DenseMatrixType DenseMatrixT>
    BlockMatrix(const DenseMatrixT& matrix, const std::vector<int>& blockDimensions)
    {
        fromDense(matrix, blockDimensions);
    }

    BlockMatrix(UnorderedElementMap&& elements, const std::vector<int>& blockDimensions)
    {
        fromUnorderedMap(std::move(elements), blockDimensions);
    }

    BlockMatrix(BlockMatrix&& other) noexcept
        : _blockDims(std::move(other._blockDims)), _elements(std::move(other._elements)),
          _nonZeroRows(std::move(other._nonZeroRows)), _nonZeroCols(std::move(other._nonZeroCols))
    {
    }

    BlockMatrix(const BlockMatrix& other) noexcept
        : _blockDims(other._blockDims), _elements(other._elements), _nonZeroRows(other._nonZeroRows),
          _nonZeroCols(other._nonZeroCols)
    {
    }

    BlockMatrix& operator=(BlockMatrix&& other)
    {
        _blockDims   = std::move(other._blockDims);
        _elements    = std::move(other._elements);
        _nonZeroRows = std::move(other._nonZeroRows);
        _nonZeroCols = std::move(other._nonZeroCols);
        return *this;
    }

    BlockMatrix& operator=(const BlockMatrix& other)
    {
        _blockDims   = other._blockDims;
        _elements    = other._elements;
        _nonZeroRows = other._nonZeroRows;
        _nonZeroCols = other._nonZeroCols;
        return *this;
    }

    BlockMatrix& operator*=(T x)
    {
        for (auto& el : _elements)
        {
            el.second *= x;
        }
        return *this;
    }

    const std::vector<int>& blockDimensions() const noexcept
    {
        return _blockDims;
    }

    ///
    /// @brief  Returns the number of blocks of each row/column, i.e., the matrix separated is into numBlocks*numBlocks blocks.
    ///
    int numBlocks() const noexcept
    {
        return static_cast<int>(_blockDims.size());
    }

    ///
    /// @brief  Returns the number of non-zero blocks.
    ///
    int size() const noexcept
    {
        return static_cast<int>(_elements.size());
    }

    int totalRows() const noexcept
    {
        int returnValue = 0;
        for (int dim : _blockDims)
        {
            returnValue += dim;
        }
        return returnValue;
    }

    int totalCols() const noexcept
    {
        return totalRows();
    }

    void fromUnorderedMap(UnorderedElementMap&& elements, const std::vector<int>& blockDimensions)
    {
        _blockDims    = blockDimensions;
        int numBlocks = this->numBlocks();

        _elements.clear();
        _nonZeroRows.clear();
        _nonZeroCols.clear();
        _nonZeroRows.resize(numBlocks);
        _nonZeroCols.resize(numBlocks);

        for (auto it = elements.begin(); it != elements.end(); ++it)
        {
            int row = it->first.first;
            int col = it->first.second;

#ifdef REAL_TIME_TRANSPORT_DEBUG
            if (it->second.rows() != _blockDims[row] || it->second.cols() != _blockDims[col])
            {
                throw Error("Block has inconsistent size");
            }
#endif

            if (it->second.isZero() == false)
            {
                _elements[{row, col}] = std::move(it->second);
                _nonZeroRows[col].insert(row);
                _nonZeroCols[row].insert(col);
            }
        }
    }

    template <typename DenseMatrixT>
    void fromDense(const DenseMatrixT& matrix, const std::vector<int>& blockDimensions)
    {
        _blockDims    = blockDimensions;
        int numBlocks = this->numBlocks();

        _elements.clear();
        _nonZeroRows.clear();
        _nonZeroCols.clear();
        _nonZeroRows.resize(numBlocks);
        _nonZeroCols.resize(numBlocks);

        // Partition the matrix into blocks and store them
        for (int i = 0; i < numBlocks; ++i)
        {
            for (int j = 0; j < numBlocks; ++j)
            {
                MatrixType block = extractDenseBlock(matrix, i, j, blockDimensions);

                if (block.isZero() == false)
                {
                    _elements[{i, j}] = std::move(block);
                    _nonZeroRows[j].insert(i);
                    _nonZeroCols[i].insert(j);
                }
            }
        }
    }

    MatrixType toDense() const
    {
        // Determine the total size of the dense matrix
        int totalRows = this->totalRows();
        int numBlocks = this->numBlocks();

        // Initialize the dense matrix with zeros
        MatrixType denseMatrix(totalRows, totalRows);
        denseMatrix.setZero();

        // Variables to keep track of the start of each block
        int startRow = 0, startCol = 0;

        // Iterate over the blocks and place them in the dense matrix
        for (int i = 0; i < numBlocks; ++i)
        {
            startCol = 0; // Reset startCol for each new row of blocks
            for (int j = 0; j < numBlocks; ++j)
            {
                // Place the block at the correct position in the dense matrix
                auto it = _elements.find({i, j});
                if (it != _elements.end())
                {
                    denseMatrix.block(startRow, startCol, _blockDims[i], _blockDims[j]) = it->second;
                }

                // Update startCol for the next block in the same row
                startCol += _blockDims[j];
            }
            // Update startRow for the next row of blocks
            startRow += _blockDims[i];
        }

        return denseMatrix;
    }

    // Get block at position (i, j)
    const MatrixType& operator()(int i, int j) const
    {
        auto it = _elements.find({i, j});

        if (it == _elements.end())
        {
            throw Error("Can not access non-existant block");
        }

        return it->second;
    }

    void addToBlock(int i, int j, MatrixType&& A)
    {
        if (A.isZero() == false)
        {
            auto it = _elements.find({i, j});
            if (it == _elements.end())
            {
                _elements[{i, j}] = std::move(A);
                _nonZeroRows[j].insert(i);
                _nonZeroCols[i].insert(j);
            }
            else
            {
                it->second += A;
                if (it->second.isZero() == true)
                {
                    _elements.erase(it);
                    _nonZeroRows[j].erase(i);
                    _nonZeroCols[i].erase(j);
                }
            }
        }
    }

    // Return true if the block at (i ,j) is not zero
    bool contains(int i, int j) const noexcept
    {
        return _elements.find({i, j}) != _elements.end();
    }

    const std::set<int>& nonZeroRows(int j) const
    {
        return _nonZeroRows[j];
    }

    const std::set<int>& nonZeroCols(int i) const
    {
        return _nonZeroCols[i];
    }

    auto begin() noexcept
    {
        return _elements.begin();
    }

    auto end() noexcept
    {
        return _elements.end();
    }

    auto begin() const noexcept
    {
        return _elements.begin();
    }

    auto end() const noexcept
    {
        return _elements.end();
    }

    auto find(int i, int j) const
    {
        return _elements.find({i, j});
    }

  private:
    std::vector<int> _blockDims;   // Dimensions of each block in the diagonal
    UnorderedElementMap _elements; // Stores the blocks

    // _nonZeroRows[j] contains indices i of rows such that the block i,j is not zero
    std::vector<std::set<int>> _nonZeroRows;

    // _nonZeroCols[i] contains indices j of columns such that the block i,j is not zero
    std::vector<std::set<int>> _nonZeroCols;
};

extern template class BlockMatrix<SciCore::Complex>;

// Computes result += alpha * A * x
// The vector blockStartIndices should be computed as
// const std::vector<int>& blockDims = A.blockDimensions();
// std::vector<int> blockStartIndices(blockDims.size(), 0);
// for (size_t i = 1; i < blockDims.size(); ++i)
// {
//     blockStartIndices[i] = blockStartIndices[i - 1] + blockDims[i - 1];
// }
template <typename ScalarT, typename T>
void addProduct(
    ScalarT alpha,
    const BlockMatrix<T>& A,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& result,
    const std::vector<int>& blockStartIndices)
{
    using MatrixType = typename BlockMatrix<T>::MatrixType;

    const std::vector<int>& blockDims = A.blockDimensions();

    // Iterate over each block
    for (const auto& element : A)
    {
        int blockRow = element.first.first;
        int blockCol = element.first.second;

        // Block starting indices
        int rowStart = blockStartIndices[blockRow];
        int colStart = blockStartIndices[blockCol];

        const MatrixType& blockMatrix = element.second;
        auto vectorSegment            = x.segment(colStart, blockDims[blockCol]);

        // Multiply and accumulate the result
        result.segment(rowStart, blockDims[blockRow]) += alpha * blockMatrix * vectorSegment;
    }
}

// Computes result += alpha * x * A
// The vector blockStartIndices should be computed as
// const std::vector<int>& blockDims = A.blockDimensions();
// std::vector<int> blockStartIndices(blockDims.size(), 0);
// for (size_t i = 1; i < blockDims.size(); ++i)
// {
//     blockStartIndices[i] = blockStartIndices[i - 1] + blockDims[i - 1];
// }
template <typename ScalarT, typename T>
void addProduct(
    ScalarT alpha,
    const Eigen::Matrix<T, 1, Eigen::Dynamic>& x,
    const BlockMatrix<T>& A,
    Eigen::Matrix<T, 1, Eigen::Dynamic>& result,
    const std::vector<int>& blockStartIndices)
{
    using MatrixType = typename BlockMatrix<T>::MatrixType;

    const std::vector<int>& blockDims = A.blockDimensions();

    // Iterate over each block
    for (const auto& element : A)
    {
        int blockRow = element.first.first;
        int blockCol = element.first.second;

        // Block starting indices
        int rowStart = blockStartIndices[blockRow];
        int colStart = blockStartIndices[blockCol];

        const MatrixType& blockMatrix = element.second;
        auto vectorSegment            = x.segment(rowStart, blockDims[blockRow]);

        // Multiply and accumulate the result
        result.segment(colStart, blockDims[blockCol]) += alpha * vectorSegment * blockMatrix;
    }
}

// Returns alpha * x * A
template <typename ScalarT, typename T>
Eigen::Matrix<T, 1, Eigen::Dynamic> product(
    ScalarT alpha,
    const Eigen::Matrix<T, 1, Eigen::Dynamic>& x,
    const BlockMatrix<T>& A,
    const std::vector<int>& blockStartIndices)
{
    using ReturnType = Eigen::Matrix<T, 1, Eigen::Dynamic>;

    ReturnType returnValue = ReturnType::Zero(x.size());
    addProduct(alpha, x, A, returnValue, blockStartIndices);

    return returnValue;
}

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_MATRIX_H
