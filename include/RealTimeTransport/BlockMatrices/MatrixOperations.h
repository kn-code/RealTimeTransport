//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_BLOCK_MATRICES_MATRIX_OPERATIONS_H
#define REAL_TIME_TRANSPORT_BLOCK_MATRICES_MATRIX_OPERATIONS_H

#include <SciCore/Utility.h>

#include "BlockDiagonalMatrix.h"
#include "BlockMatrix.h"
#include "BlockVector.h"

namespace RealTimeTransport
{

// Computes result += α * \sum_k A_{ik} B_{kj}, where i,j,k refer to block indices for fixed i,j
template <typename ScalarT, typename T>
void addProduct(
    int i,
    int j,
    ScalarT alpha,
    const BlockMatrix<T>& A,
    const BlockMatrix<T>& B,
    typename BlockMatrix<T>::MatrixType& result)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (A.blockDimensions() != B.blockDimensions())
    {
        throw Error("Block dimensions don't match");
    }
#endif

    foreachCommonElement(
        [&](int k)
        {
#ifdef REAL_TIME_TRANSPORT_DEBUG
            if (result.rows() != A(i, k).rows() || result.cols() != B(k, j).cols())
            {
                throw Error("Block dimensions don't match");
            }
#endif
            result.noalias() += alpha * A(i, k) * B(k, j);
        },
        A.nonZeroCols(i), B.nonZeroRows(j));
}

// Returns α * \sum_k A_{ik} B_{kj}, where i,j,k refer to block indices for fixed i,j
template <typename ScalarT, typename T>
typename BlockMatrix<T>::MatrixType product(
    int i,
    int j,
    ScalarT alpha,
    const BlockMatrix<T>& A,
    const BlockMatrix<T>& B)
{
    using ReturnType = typename BlockMatrix<T>::MatrixType;

    ReturnType returnValue = ReturnType::Zero(A.blockDimensions()[i], B.blockDimensions()[j]);
    addProduct(i, j, alpha, A, B, returnValue);

    return returnValue;
}

// Computed result += α * A * B * C
template <typename ScalarT, typename T>
void addProduct(
    ScalarT alpha,
    const BlockMatrix<T>& A,
    const BlockMatrix<T>& B,
    const BlockMatrix<T>& C,
    BlockMatrix<T>& result)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (A.blockDimensions() != B.blockDimensions() || A.blockDimensions() != result.blockDimensions())
    {
        throw Error("Block dimensions don't match");
    }
#endif

    using MatrixType = typename BlockMatrix<T>::MatrixType;
    int numBlocks    = A.numBlocks();

    // A_ni B_ij C_jm
    for (int n = 0; n < numBlocks; ++n)
    {
        for (int i : A.nonZeroCols(n))
        {
            for (int j : B.nonZeroCols(i))
            {
                for (int m : C.nonZeroCols(j))
                {
                    MatrixType ABC_nm = alpha * A(n, i) * B(i, j) * C(j, m);
                    result.addToBlock(n, m, std::move(ABC_nm));
                }
            }
        }
    }
}

template <typename ScalarT, typename T>
BlockMatrix<T> product(ScalarT alpha, const BlockMatrix<T>& A, const BlockMatrix<T>& B, const BlockMatrix<T>& C)
{
    BlockMatrix<T> returnValue(A.blockDimensions());
    addProduct(alpha, A, B, C, returnValue);
    return returnValue;
}

// Computes result += α * A * B. It is assumed that A and B are such that the result is block diagonal.
template <typename ScalarT, typename T>
void addProduct(ScalarT alpha, const BlockMatrix<T>& A, const BlockMatrix<T>& B, BlockDiagonalMatrix& result)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (A.blockDimensions() != B.blockDimensions() || A.blockDimensions() != result.blockDimensions())
    {
        throw Error("Block dimensions don't match");
    }
#endif

    // Iterate over the blocks
    int numBlocks = A.numBlocks();
    for (int i = 0; i < numBlocks; ++i)
    {
        addProduct(i, i, alpha, A, B, result(i));
    }

#ifdef REAL_TIME_TRANSPORT_DEBUG
    for (int i = 0; i < numBlocks; ++i)
    {
        for (int j = 0; j < numBlocks; ++j)
        {
            if (i != j)
            {
                if (product(i, j, alpha, A, B).isZero() == false)
                {
                    throw Error("Result is not block diagonal as required.");
                }
            }
        }
    }
#endif
}

template <typename ScalarT, typename T>
BlockDiagonalMatrix product_toDiagonal(ScalarT alpha, const BlockMatrix<T>& A, const BlockMatrix<T>& B)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (A.blockDimensions() != B.blockDimensions())
    {
        throw Error("Block dimensions don't match");
    }
#endif

    BlockDiagonalMatrix returnValue = BlockDiagonalMatrix::Zero(A.blockDimensions());
    addProduct(alpha, A, B, returnValue);
    return returnValue;
}

// Computes result += α \sum_i A1_{ni} D_i A2_{im} where n,m,i refer to block indices and n,m is fixed
template <typename ScalarT, typename T>
void addProduct(
    int n,
    int m,
    ScalarT alpha,
    const BlockMatrix<T>& A1,
    const BlockDiagonalMatrix& D,
    const BlockMatrix<T>& A2,
    typename BlockMatrix<T>::MatrixType& result)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (A1.blockDimensions() != A2.blockDimensions())
    {
        throw Error("Block dimensions don't match");
    }
#endif

    foreachCommonElement(
        [&](int i) { result.noalias() += alpha * A1(n, i) * D(i) * A2(i, m); }, A1.nonZeroCols(n), A2.nonZeroRows(m));
}

// Computes result += α \sum_i A1_{ni} A2_{ij} A3_{jm}
template <typename ScalarT, typename T>
void addProduct(
    int n,
    int m,
    ScalarT alpha,
    const BlockMatrix<T>& A1,
    const BlockMatrix<T>& A2,
    const BlockMatrix<T>& A3,
    typename BlockMatrix<T>::MatrixType& result)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (A1.blockDimensions() != A2.blockDimensions() || A2.blockDimensions() != A3.blockDimensions())
    {
        throw Error("Block dimensions don't match");
    }
#endif

    for (int i : A1.nonZeroCols(n))
    {
        foreachCommonElement(
            [&](int j) { result.noalias() += alpha * A1(n, i) * A2(i, j) * A3(j, m); }, A2.nonZeroCols(i),
            A3.nonZeroRows(m));
    }
}

// Computes result += α \sum_i A_{ni} D_{i} x_i
template <typename ScalarT, typename T>
void addProduct(
    int n,
    ScalarT alpha,
    const BlockMatrix<T>& A,
    const BlockDiagonalMatrix& D,
    const BlockVector<T>& x,
    typename BlockMatrix<T>::MatrixType& result)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (A.blockDimensions() != D.blockDimensions())
    {
        throw Error("Block dimensions don't match");
    }
#endif

    for (int i : A.nonZeroCols(n))
    {
        auto it_x_i = x.find(i);
        if (it_x_i != x.end())
        {
#ifdef REAL_TIME_TRANSPORT_DEBUG
            if (result.rows() != A(n, i).rows() || A(n, i).cols() != D(i).rows() || D(i).rows() != D(i).cols() ||
                D(i).cols() != it_x_i->second.rows() || it_x_i->second.cols() != result.cols())
            {
                throw Error("Dimension mismatch");
            }
#endif
            result.noalias() += alpha * A(n, i) * D(i) * it_x_i->second;
        }
    }
}

// Computes result_n += α A1_ni D_i A2_im
template <typename ScalarT, typename T>
void addProduct_col(
    int m,
    ScalarT alpha,
    const BlockMatrix<T>& A1,
    const BlockDiagonalMatrix& D,
    const BlockMatrix<T>& A2,
    BlockVector<T>& result)
{
    using MatrixType = typename BlockMatrix<T>::MatrixType;
    int numBlocks    = A1.numBlocks();

    // A1_ni D_i A2_im
    for (int n = 0; n < numBlocks; ++n)
    {
        foreachCommonElement(
            [&](int i)
            {
#ifdef REAL_TIME_TRANSPORT_DEBUG
                if (A1(n, i).cols() != D(i).rows() || D(i).rows() != D(i).cols() || D(i).cols() != A2(i, m).rows())
                {
                    throw Error("Dimension mismatch");
                }
#endif
                MatrixType ADA_n = alpha * A1(n, i) * D(i) * A2(i, m);
                result.addToBlock(n, std::move(ADA_n));
            },
            A1.nonZeroCols(n), A2.nonZeroRows(m));
    }
}

// Computes result_n += α A1_{ni} A2_{ij} A3_{jm}
// Only the vector elements in result are changed which initially exist!
template <typename ScalarT, typename T>
void addProduct_col_unsafe(
    int m,
    ScalarT alpha,
    const BlockMatrix<T>& A1,
    const BlockMatrix<T>& A2,
    const BlockMatrix<T>& A3,
    BlockVector<T>& result)
{
    for (auto it = result.begin(); it != result.end(); ++it)
    {
        int n = it->first;
        addProduct(n, m, alpha, A1, A2, A3, it->second);
    }
}

// Computes result += α A D
template <typename ScalarT, typename T>
void addProduct(ScalarT alpha, const BlockMatrix<T>& A, const BlockDiagonalMatrix& D, BlockMatrix<T>& result)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (A.blockDimensions() != D.blockDimensions())
    {
        throw Error("Block dimensions don't match");
    }
#endif

    for (int i = 0; i < A.numBlocks(); ++i)
    {
        for (int j : A.nonZeroCols(i))
        {
            result.addToBlock(i, j, alpha * A(i, j) * D(j));
        }
    }
}

// Computes result_n += α A_nm D_m
template <typename ScalarT, typename T>
void addProduct_col_unsafe_2(
    int m,
    ScalarT alpha,
    const BlockMatrix<T>& A,
    const BlockDiagonalMatrix& D,
    BlockVector<T>& result)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (A.blockDimensions() != D.blockDimensions())
    {
        throw Error("Block dimensions don't match");
    }
#endif

    for (auto it = result.begin(); it != result.end(); ++it)
    {
        int n    = it->first;
        auto itA = A.find(n, m);
        if (itA != A.end())
        {
#ifdef REAL_TIME_TRANSPORT_DEBUG
            if (it->second.rows() != itA->second.rows() || itA->second.cols() != D(m).rows() ||
                D(m).cols() != it->second.cols())
            {
                throw Error("Dimension mismatch");
            }
#endif
            it->second.noalias() += alpha * itA->second * D(m);
        }
    }
}

// Computes A -> D1 A D2
template <typename T>
void productCombination_1(const BlockDiagonalMatrix& D1, BlockMatrix<T>& A, const BlockDiagonalMatrix& D2)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (D1.blockDimensions() != A.blockDimensions() || A.blockDimensions() != D2.blockDimensions())
    {
        throw Error("Block dimensions don't match");
    }
#endif

    for (auto it = A.begin(); it != A.end(); ++it)
    {
        it->second = D1(it->first.first) * it->second * D2(it->first.second);
    }
}

// Computes A -> D1 A D2 + D1 A + A D2
template <typename T>
void productCombination_2(const BlockDiagonalMatrix& D1, BlockMatrix<T>& A, const BlockDiagonalMatrix& D2)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (D1.blockDimensions() != A.blockDimensions() || A.blockDimensions() != D2.blockDimensions())
    {
        throw Error("Block dimensions don't match");
    }
#endif

    for (auto it = A.begin(); it != A.end(); ++it)
    {
        int i      = it->first.first;
        int j      = it->first.second;
        it->second = D1(i) * it->second * D2(j) + D1(i) * it->second + it->second * D2(j);
    }
}

// Returns α D1 D2 D3
template <typename ScalarT>
BlockDiagonalMatrix product(
    ScalarT alpha,
    const BlockDiagonalMatrix& D1,
    const BlockDiagonalMatrix& D2,
    const BlockDiagonalMatrix& D3)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (D1.blockDimensions() != D2.blockDimensions() || D2.blockDimensions() != D3.blockDimensions())
    {
        throw Error("Block dimensions don't match");
    }
#endif

    auto returnValue = BlockDiagonalMatrix::Zero(D1.blockDimensions());

    for (int i = 0; i < D1.numBlocks(); ++i)
    {
        returnValue(i).noalias() = alpha * D1(i) * D2(i) * D3(i);
    }

    return returnValue;
}

// Computes result += alpha * x * D * A
template <typename ScalarT, typename T>
void addProduct(
    ScalarT alpha,
    const Eigen::Matrix<T, 1, Eigen::Dynamic>& x,
    const BlockDiagonalMatrix& D,
    const BlockMatrix<T>& A,
    Eigen::Matrix<T, 1, Eigen::Dynamic>& result,
    const std::vector<int>& blockStartIndices)
{
    Eigen::Matrix<T, 1, Eigen::Dynamic> tmp = product(alpha, x, D);
    addProduct(1.0, tmp, A, result, blockStartIndices);
}

// Computes result += α \sum_i A1_{ni} A2_{ij} x_j
template <typename ScalarT, typename T>
void addProduct(
    int n,
    ScalarT alpha,
    const BlockMatrix<T>& A1,
    const BlockMatrix<T>& A2,
    const BlockVector<T>& x,
    typename BlockMatrix<T>::MatrixType& result)
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (A1.blockDimensions() != A2.blockDimensions())
    {
        throw Error("Block dimensions don't match");
    }
#endif

    for (int i : A1.nonZeroCols(n))
    {
        for (int j : A2.nonZeroCols(i))
        {
            auto it_x_j = x.find(j);
            if (it_x_j != x.end())
            {
#ifdef REAL_TIME_TRANSPORT_DEBUG
                if (result.rows() != A1(n, i).rows() || A1(n, i).cols() != A2(i, j).rows() ||
                    A2(i, j).cols() != it_x_j->second.rows() || it_x_j->second.cols() != result.cols())
                {
                    throw Error("Dimension mismatch");
                }
#endif
                result.noalias() += alpha * A1(n, i) * A2(i, j) * it_x_j->second;
            }
        }
    }
}

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_BLOCK_MATRICES_MATRIX_OPERATIONS_H
