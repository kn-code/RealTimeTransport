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
class REALTIMETRANSPORT_EXPORT BlockMatrix
{
  public:
    using Scalar              = SciCore::Complex;
    using MatrixType          = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using UnorderedElementMap = boost::unordered_flat_map<std::pair<int, int>, MatrixType>;

    BlockMatrix() noexcept;
    BlockMatrix(const std::vector<int>& blockDimensions);

    template <SciCore::DenseMatrixType DenseMatrixT>
    BlockMatrix(const DenseMatrixT& matrix, const std::vector<int>& blockDimensions)
    {
        fromDense(matrix, blockDimensions);
    }

    BlockMatrix(UnorderedElementMap&& elements, const std::vector<int>& blockDimensions);

    BlockMatrix(BlockMatrix&& other) noexcept;
    BlockMatrix(const BlockMatrix& other) noexcept;

    BlockMatrix& operator=(BlockMatrix&& other);
    BlockMatrix& operator=(const BlockMatrix& other);

    BlockMatrix& operator*=(Scalar x);
    BlockMatrix& operator+=(const BlockMatrix& other);

    const std::vector<int>& blockDimensions() const noexcept;

    ///
    /// @brief  Returns the number of blocks of each row/column, i.e., the matrix separated is into numBlocks*numBlocks blocks.
    ///
    int numBlocks() const noexcept;

    ///
    /// @brief  Returns the number of non-zero blocks.
    ///
    int size() const noexcept;

    int totalRows() const noexcept;

    int totalCols() const noexcept;

    void fromUnorderedMap(UnorderedElementMap&& elements, const std::vector<int>& blockDimensions);

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

    MatrixType toDense() const;

    // Get block at position (i, j)
    const MatrixType& operator()(int i, int j) const;

    void addToBlock(int i, int j, MatrixType&& A);

    // Return true if the block at (i ,j) is not zero
    bool contains(int i, int j) const noexcept;

    const std::set<int>& nonZeroRows(int j) const;
    const std::set<int>& nonZeroCols(int i) const;

    UnorderedElementMap::iterator begin() noexcept;
    UnorderedElementMap::iterator end() noexcept;
    UnorderedElementMap::const_iterator begin() const noexcept;
    UnorderedElementMap::const_iterator end() const noexcept;

    UnorderedElementMap::const_iterator find(int i, int j) const;

  private:
    std::vector<int> _blockDims;   // Dimensions of each block in the diagonal
    UnorderedElementMap _elements; // Stores the blocks

    // _nonZeroRows[j] contains indices i of rows such that the block i,j is not zero
    std::vector<std::set<int>> _nonZeroRows;

    // _nonZeroCols[i] contains indices j of columns such that the block i,j is not zero
    std::vector<std::set<int>> _nonZeroCols;
};

// Computes result += alpha * A * x
// The vector blockStartIndices should be computed as
// const std::vector<int>& blockDims = A.blockDimensions();
// std::vector<int> blockStartIndices(blockDims.size(), 0);
// for (size_t i = 1; i < blockDims.size(); ++i)
// {
//     blockStartIndices[i] = blockStartIndices[i - 1] + blockDims[i - 1];
// }
REALTIMETRANSPORT_EXPORT void addProduct(
    SciCore::Complex alpha,
    const BlockMatrix& A,
    const Eigen::Matrix<SciCore::Complex, Eigen::Dynamic, 1>& x,
    Eigen::Matrix<SciCore::Complex, Eigen::Dynamic, 1>& result,
    const std::vector<int>& blockStartIndices);

REALTIMETRANSPORT_EXPORT void addProduct(
    SciCore::Complex alpha,
    const Eigen::Matrix<SciCore::Complex, 1, Eigen::Dynamic>& x,
    const BlockMatrix& A,
    Eigen::Matrix<SciCore::Complex, 1, Eigen::Dynamic>& result,
    const std::vector<int>& blockStartIndices);

// Returns alpha * x * A
REALTIMETRANSPORT_EXPORT Eigen::Matrix<SciCore::Complex, 1, Eigen::Dynamic> product(
    SciCore::Complex alpha,
    const Eigen::Matrix<SciCore::Complex, 1, Eigen::Dynamic>& x,
    const BlockMatrix& A,
    const std::vector<int>& blockStartIndices);

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_MATRIX_H
