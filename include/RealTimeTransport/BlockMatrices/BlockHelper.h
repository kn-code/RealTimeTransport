//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_HELPER_H
#define REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_HELPER_H

#include <set>
#include <vector>

#include <SciCore/Definitions.h>

namespace RealTimeTransport
{

template <typename FuncT>
void foreachCommonElement(FuncT&& f, const std::set<int>& x, const std::set<int>& y)
{
    auto it_x = x.begin();
    auto it_y = y.begin();

    while (it_x != x.end() && it_y != y.end())
    {
        if (*it_x < *it_y)
        {
            ++it_x;
        }
        else if (*it_y < *it_x)
        {
            ++it_y;
        }
        else
        {
            // Found a common element
            f(*it_x);
            ++it_x;
            ++it_y;
        }
    }
}

// Helper function to extract the block indexed by i, j from a dense matrix
template <typename MatrixT>
MatrixT extractDenseBlock(const MatrixT& A, int i, int j, const std::vector<int>& blockDims)
{
    // Calculate the start row and column
    int startRow = 0;
    for (int k = 0; k < i; ++k)
    {
        startRow += blockDims[k];
    }

    int startCol = 0;
    for (int k = 0; k < j; ++k)
    {
        startCol += blockDims[k];
    }

    // Determine the size of the block
    int numRows = blockDims[i];
    int numCols = blockDims[j];

    // Extract and return the block
    return A.block(startRow, startCol, numRows, numCols);
}

template <typename MatrixT>
bool isBlockDiagonal(const MatrixT& matrix, const std::vector<int>& blockDimensions)
{
    int currentRow = 0;
    int numBlocks  = static_cast<int>(blockDimensions.size());
    for (int i = 0; i < numBlocks; ++i)
    {
        int currentCol = 0;
        for (int j = 0; j < numBlocks; ++j)
        {
            // If not the same block (diagonal block), check if the block is zero
            if (i != j)
            {
                auto block = matrix.block(currentRow, currentCol, blockDimensions[i], blockDimensions[j]);

                // If any element in the off-diagonal block is non-zero, return false
                if (block.isZero() == false)
                {
                    return false;
                }
            }
            currentCol += blockDimensions[j];
        }
        currentRow += blockDimensions[i];
    }
    return true;
}

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_HELPER_H
