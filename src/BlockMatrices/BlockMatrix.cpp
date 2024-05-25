//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/BlockMatrices/BlockMatrix.h"

namespace RealTimeTransport
{

BlockMatrix::BlockMatrix() noexcept
{
}

BlockMatrix::BlockMatrix(const std::vector<int>& blockDimensions) : _blockDims(blockDimensions)
{
    int numBlocks = this->numBlocks();
    _nonZeroRows.resize(numBlocks);
    _nonZeroCols.resize(numBlocks);
}

BlockMatrix::BlockMatrix(UnorderedElementMap&& elements, const std::vector<int>& blockDimensions)
{
    fromUnorderedMap(std::move(elements), blockDimensions);
}

BlockMatrix::BlockMatrix(BlockMatrix&& other) noexcept
    : _blockDims(std::move(other._blockDims)), _elements(std::move(other._elements)),
      _nonZeroRows(std::move(other._nonZeroRows)), _nonZeroCols(std::move(other._nonZeroCols))
{
}

BlockMatrix::BlockMatrix(const BlockMatrix& other) noexcept
    : _blockDims(other._blockDims), _elements(other._elements), _nonZeroRows(other._nonZeroRows),
      _nonZeroCols(other._nonZeroCols)
{
}

BlockMatrix& BlockMatrix::operator=(BlockMatrix&& other)
{
    _blockDims   = std::move(other._blockDims);
    _elements    = std::move(other._elements);
    _nonZeroRows = std::move(other._nonZeroRows);
    _nonZeroCols = std::move(other._nonZeroCols);
    return *this;
}

BlockMatrix& BlockMatrix::operator=(const BlockMatrix& other)
{
    _blockDims   = other._blockDims;
    _elements    = other._elements;
    _nonZeroRows = other._nonZeroRows;
    _nonZeroCols = other._nonZeroCols;
    return *this;
}

BlockMatrix& BlockMatrix::operator*=(BlockMatrix::Scalar x)
{
    for (auto& el : _elements)
    {
        el.second *= x;
    }
    return *this;
}

const std::vector<int>& BlockMatrix::blockDimensions() const noexcept
{
    return _blockDims;
}

int BlockMatrix::numBlocks() const noexcept
{
    return static_cast<int>(_blockDims.size());
}

int BlockMatrix::size() const noexcept
{
    return static_cast<int>(_elements.size());
}

int BlockMatrix::totalRows() const noexcept
{
    int returnValue = 0;
    for (int dim : _blockDims)
    {
        returnValue += dim;
    }
    return returnValue;
}

int BlockMatrix::totalCols() const noexcept
{
    return totalRows();
}

void BlockMatrix::fromUnorderedMap(UnorderedElementMap&& elements, const std::vector<int>& blockDimensions)
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

BlockMatrix::MatrixType BlockMatrix::toDense() const
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
const BlockMatrix::MatrixType& BlockMatrix::operator()(int i, int j) const
{
    auto it = _elements.find({i, j});

    if (it == _elements.end())
    {
        throw Error("Can not access non-existant block");
    }

    return it->second;
}

void BlockMatrix::addToBlock(int i, int j, BlockMatrix::MatrixType&& A)
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
bool BlockMatrix::contains(int i, int j) const noexcept
{
    return _elements.find({i, j}) != _elements.end();
}

const std::set<int>& BlockMatrix::nonZeroRows(int j) const
{
    return _nonZeroRows[j];
}

const std::set<int>& BlockMatrix::nonZeroCols(int i) const
{
    return _nonZeroCols[i];
}

BlockMatrix::UnorderedElementMap::iterator BlockMatrix::begin() noexcept
{
    return _elements.begin();
}

BlockMatrix::UnorderedElementMap::iterator BlockMatrix::end() noexcept
{
    return _elements.end();
}

BlockMatrix::UnorderedElementMap::const_iterator BlockMatrix::begin() const noexcept
{
    return _elements.begin();
}

BlockMatrix::UnorderedElementMap::const_iterator BlockMatrix::end() const noexcept
{
    return _elements.end();
}

BlockMatrix::UnorderedElementMap::const_iterator BlockMatrix::find(int i, int j) const
{
    return _elements.find({i, j});
}

void addProduct(
    SciCore::Complex alpha,
    const BlockMatrix& A,
    const Eigen::Matrix<SciCore::Complex, Eigen::Dynamic, 1>& x,
    Eigen::Matrix<SciCore::Complex, Eigen::Dynamic, 1>& result,
    const std::vector<int>& blockStartIndices)
{
    using MatrixType = typename BlockMatrix::MatrixType;

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

void addProduct(
    SciCore::Complex alpha,
    const Eigen::Matrix<SciCore::Complex, 1, Eigen::Dynamic>& x,
    const BlockMatrix& A,
    Eigen::Matrix<SciCore::Complex, 1, Eigen::Dynamic>& result,
    const std::vector<int>& blockStartIndices)
{
    using MatrixType = typename BlockMatrix::MatrixType;

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

Eigen::Matrix<SciCore::Complex, 1, Eigen::Dynamic> product(
    SciCore::Complex alpha,
    const Eigen::Matrix<SciCore::Complex, 1, Eigen::Dynamic>& x,
    const BlockMatrix& A,
    const std::vector<int>& blockStartIndices)
{
    using ReturnType = Eigen::Matrix<SciCore::Complex, 1, Eigen::Dynamic>;

    ReturnType returnValue = ReturnType::Zero(x.size());
    addProduct(alpha, x, A, returnValue, blockStartIndices);

    return returnValue;
}

} // namespace RealTimeTransport
