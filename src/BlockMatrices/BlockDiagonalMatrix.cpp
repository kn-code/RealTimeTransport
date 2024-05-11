//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include <SciCore/Utility.h>

#include "RealTimeTransport/BlockMatrices/BlockDiagonalMatrix.h"

namespace RealTimeTransport
{

BlockDiagonalMatrix::BlockDiagonalMatrix() noexcept
{
}

BlockDiagonalMatrix::BlockDiagonalMatrix(std::vector<MatrixType>&& newBlocks) noexcept
{
    fromBlocks(std::move(newBlocks));
}

BlockDiagonalMatrix::BlockDiagonalMatrix(BlockDiagonalMatrix&& other) noexcept : _blocks(std::move(other._blocks))
{
}

BlockDiagonalMatrix::BlockDiagonalMatrix(const BlockDiagonalMatrix& other) : _blocks(other._blocks)
{
}

bool BlockDiagonalMatrix::operator==(const BlockDiagonalMatrix& other) const
{
    return _blocks == other._blocks;
}

bool BlockDiagonalMatrix::operator!=(const BlockDiagonalMatrix& other) const
{
    return !operator==(other);
}

BlockDiagonalMatrix BlockDiagonalMatrix::Zero(const std::vector<int>& blockDimensions)
{
    BlockDiagonalMatrix returnValue;
    returnValue.setZero(blockDimensions);
    return returnValue;
}

BlockDiagonalMatrix BlockDiagonalMatrix::Identity(const std::vector<int>& blockDimensions)
{
    size_t numBlocks = blockDimensions.size();
    std::vector<MatrixType> blocks(numBlocks);
    for (size_t i = 0; i < numBlocks; ++i)
    {
        blocks[i] = MatrixType::Identity(blockDimensions[i], blockDimensions[i]);
    }
    return BlockDiagonalMatrix(std::move(blocks));
}

BlockDiagonalMatrix& BlockDiagonalMatrix::operator=(BlockDiagonalMatrix&& other) noexcept
{
    _blocks = std::move(other._blocks);

    return *this;
}

BlockDiagonalMatrix& BlockDiagonalMatrix::operator=(const BlockDiagonalMatrix& other)
{
    _blocks = other._blocks;

    return *this;
}

BlockDiagonalMatrix& BlockDiagonalMatrix::operator+=(const BlockDiagonalMatrix& rhs)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (blockDimensions() != rhs.blockDimensions())
    {
        throw Error("Block dimensions must be equal");
    }
#endif
    for (size_t i = 0; i < _blocks.size(); ++i)
    {
        _blocks[i] += rhs._blocks[i];
    }

    return *this;
}

BlockDiagonalMatrix& BlockDiagonalMatrix::operator-=(const BlockDiagonalMatrix& rhs)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (blockDimensions() != rhs.blockDimensions())
    {
        throw Error("Block dimensions must be equal");
    }
#endif
    for (size_t i = 0; i < _blocks.size(); ++i)
    {
        _blocks[i] -= rhs._blocks[i];
    }

    return *this;
}

BlockDiagonalMatrix& BlockDiagonalMatrix::operator*=(const BlockDiagonalMatrix& rhs)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (blockDimensions() != rhs.blockDimensions())
    {
        throw Error("Block dimensions must be equal");
    }
#endif
    for (size_t i = 0; i < _blocks.size(); ++i)
    {
        _blocks[i] *= rhs._blocks[i];
    }

    return *this;
}

BlockDiagonalMatrix& BlockDiagonalMatrix::operator*=(SciCore::Real scalar)
{
    for (size_t i = 0; i < _blocks.size(); ++i)
    {
        _blocks[i] *= scalar;
    }

    return *this;
}

BlockDiagonalMatrix& BlockDiagonalMatrix::operator*=(SciCore::Complex scalar)
{
    for (size_t i = 0; i < _blocks.size(); ++i)
    {
        _blocks[i] *= scalar;
    }

    return *this;
}

int BlockDiagonalMatrix::numBlocks() const noexcept
{
    return static_cast<int>(_blocks.size());
}

int BlockDiagonalMatrix::totalRows() const noexcept
{
    int returnValue = 0;
    for (const auto& block : _blocks)
    {
        returnValue += block.rows();
    }
    return returnValue;
}

int BlockDiagonalMatrix::totalCols() const noexcept
{
    return totalRows(); // Matrix always quadratic
}

std::vector<int> BlockDiagonalMatrix::blockDimensions() const
{
    std::vector<int> returnValue(_blocks.size());
    for (size_t i = 0; i < _blocks.size(); ++i)
    {
        returnValue[i] = _blocks[i].rows();
    }
    return returnValue;
}

BlockDiagonalMatrix::MatrixType& BlockDiagonalMatrix::operator()(int i)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (i < 0 || i >= numBlocks())
    {
        throw Error("Invalid block index");
    }
#endif

    return _blocks[i];
}

const BlockDiagonalMatrix::MatrixType& BlockDiagonalMatrix::operator()(int i) const
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (i < 0 || i >= numBlocks())
    {
        throw Error("Invalid block index");
    }
#endif

    return _blocks[i];
}

void BlockDiagonalMatrix::fromBlocks(std::vector<BlockDiagonalMatrix::MatrixType>&& newBlocks)
{
    _blocks = std::move(newBlocks);
}

void BlockDiagonalMatrix::setZero(const std::vector<int>& blockDimensions)
{
    size_t numBlocks = blockDimensions.size();
    _blocks.resize(numBlocks);

    for (size_t i = 0; i < numBlocks; ++i)
    {
        int dim    = blockDimensions[i];
        _blocks[i] = MatrixType::Zero(dim, dim);
    }
}

BlockDiagonalMatrix::MatrixType BlockDiagonalMatrix::toDense() const
{
    int rows = totalRows();
    MatrixType denseMatrix(rows, rows);
    denseMatrix.setZero();

    // Iterate over the blocks and place them in the dense matrix
    int startRow = 0;
    for (size_t i = 0; i < _blocks.size(); ++i)
    {
        denseMatrix.block(startRow, startRow, _blocks[i].rows(), _blocks[i].rows()) = _blocks[i];

        // Update startRow for the next row of blocks
        startRow += _blocks[i].rows();
    }

    return denseMatrix;
}

// Member function to return the element at a given row and column
const BlockDiagonalMatrix::Scalar& BlockDiagonalMatrix::element(int row, int col) const
{
    // Iterate through each block
    int blockStart = 0;
    for (size_t i = 0; i < _blocks.size(); ++i)
    {
        const auto& block = _blocks[i];
        int blockSize     = block.rows();
        int blockEnd      = blockStart + blockSize - 1;

        // Check if (row, col) is within the current block
        if (row >= blockStart && row <= blockEnd && col >= blockStart && col <= blockEnd)
        {
            // Calculate local indices within the block
            int localRow = row - blockStart;
            int localCol = col - blockStart;
            // Return the corresponding element from the block
            return block(localRow, localCol);
        }

        blockStart += blockSize;
    }

    // If the position is not within any block, throw an error
    throw Error("Matrix element is not within any block");
}

BlockDiagonalMatrix::Scalar& BlockDiagonalMatrix::element(int row, int col)
{
    return const_cast<Scalar&>(const_cast<const BlockDiagonalMatrix*>(this)->element(row, col));
}

SciCore::Real maxNorm(const BlockDiagonalMatrix& A)
{
    using namespace SciCore;

    Real returnValue = 0;
    for (int i = 0; i < A.numBlocks(); ++i)
    {
        Real norm = SciCore::maxNorm(A(i));
        if (norm > returnValue)
        {
            returnValue = norm;
        }
    }
    return returnValue;
}

BlockDiagonalMatrix operator+(const BlockDiagonalMatrix& lhs, const BlockDiagonalMatrix& rhs)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (lhs.blockDimensions() != rhs.blockDimensions())
    {
        throw Error("Block dimensions must be equal");
    }
#endif

    BlockDiagonalMatrix returnValue(lhs);
    returnValue += rhs;
    return returnValue;
}

BlockDiagonalMatrix operator-(const BlockDiagonalMatrix& lhs, const BlockDiagonalMatrix& rhs)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (lhs.blockDimensions() != rhs.blockDimensions())
    {
        throw Error("Block dimensions must be equal");
    }
#endif

    BlockDiagonalMatrix returnValue(lhs);
    returnValue -= rhs;
    return returnValue;
}

BlockDiagonalMatrix operator*(const BlockDiagonalMatrix& lhs, const BlockDiagonalMatrix& rhs)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (lhs.blockDimensions() != rhs.blockDimensions())
    {
        throw Error("Block dimensions must be equal");
    }
#endif

    BlockDiagonalMatrix returnValue(lhs);
    returnValue *= rhs;
    return returnValue;
}

BlockDiagonalMatrix operator*(const BlockDiagonalMatrix& lhs, SciCore::Real scalar)
{
    BlockDiagonalMatrix returnValue(lhs);
    returnValue *= scalar;
    return returnValue;
}

BlockDiagonalMatrix operator*(const BlockDiagonalMatrix& lhs, SciCore::Complex scalar)
{
    BlockDiagonalMatrix returnValue(lhs);
    returnValue *= scalar;
    return returnValue;
}

BlockDiagonalMatrix operator*(SciCore::Real scalar, const BlockDiagonalMatrix& rhs)
{
    BlockDiagonalMatrix returnValue(rhs);
    returnValue *= scalar;
    return returnValue;
}

BlockDiagonalMatrix operator*(SciCore::Complex scalar, const BlockDiagonalMatrix& rhs)
{
    BlockDiagonalMatrix returnValue(rhs);
    returnValue *= scalar;
    return returnValue;
}

} // namespace RealTimeTransport
