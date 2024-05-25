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
REALTIMETRANSPORT_EXPORT void addProduct(
    int i,
    int j,
    SciCore::Complex alpha,
    const BlockMatrix& A,
    const BlockMatrix& B,
    typename BlockMatrix::MatrixType& result);

// Returns α * \sum_k A_{ik} B_{kj}, where i,j,k refer to block indices for fixed i,j
REALTIMETRANSPORT_EXPORT typename BlockMatrix::MatrixType product(
    int i,
    int j,
    SciCore::Complex alpha,
    const BlockMatrix& A,
    const BlockMatrix& B);

// Computed result += α * A * B * C
REALTIMETRANSPORT_EXPORT void addProduct(
    SciCore::Complex alpha,
    const BlockMatrix& A,
    const BlockMatrix& B,
    const BlockMatrix& C,
    BlockMatrix& result);

REALTIMETRANSPORT_EXPORT BlockMatrix
product(SciCore::Complex alpha, const BlockMatrix& A, const BlockMatrix& B, const BlockMatrix& C);

// Computes result += α * A * B. It is assumed that A and B are such that the result is block diagonal.
REALTIMETRANSPORT_EXPORT void addProduct(
    SciCore::Complex alpha,
    const BlockMatrix& A,
    const BlockMatrix& B,
    BlockDiagonalMatrix& result);

REALTIMETRANSPORT_EXPORT BlockDiagonalMatrix
product_toDiagonal(SciCore::Complex alpha, const BlockMatrix& A, const BlockMatrix& B);

// Computes result += α \sum_i A1_{ni} D_i A2_{im} where n,m,i refer to block indices and n,m is fixed
REALTIMETRANSPORT_EXPORT void addProduct(
    int n,
    int m,
    SciCore::Complex alpha,
    const BlockMatrix& A1,
    const BlockDiagonalMatrix& D,
    const BlockMatrix& A2,
    typename BlockMatrix::MatrixType& result);

// Computes result += α \sum_i A1_{ni} A2_{ij} A3_{jm}
REALTIMETRANSPORT_EXPORT void addProduct(
    int n,
    int m,
    SciCore::Complex alpha,
    const BlockMatrix& A1,
    const BlockMatrix& A2,
    const BlockMatrix& A3,
    typename BlockMatrix::MatrixType& result);

// Computes result += α \sum_i A_{ni} D_{i} x_i
REALTIMETRANSPORT_EXPORT void addProduct(
    int n,
    SciCore::Complex alpha,
    const BlockMatrix& A,
    const BlockDiagonalMatrix& D,
    const BlockVector& x,
    typename BlockMatrix::MatrixType& result);

// Computes result_n += α A1_ni D_i A2_im
REALTIMETRANSPORT_EXPORT void addProduct_col(
    int m,
    SciCore::Complex alpha,
    const BlockMatrix& A1,
    const BlockDiagonalMatrix& D,
    const BlockMatrix& A2,
    BlockVector& result);

// Computes result_n += α A1_{ni} A2_{ij} A3_{jm}
// Only the vector elements in result are changed which initially exist!
REALTIMETRANSPORT_EXPORT void addProduct_col_unsafe(
    int m,
    SciCore::Complex alpha,
    const BlockMatrix& A1,
    const BlockMatrix& A2,
    const BlockMatrix& A3,
    BlockVector& result);

// Computes result += α A D
REALTIMETRANSPORT_EXPORT void addProduct(
    SciCore::Complex alpha,
    const BlockMatrix& A,
    const BlockDiagonalMatrix& D,
    BlockMatrix& result);

// Computes result_n += α A_nm D_m
REALTIMETRANSPORT_EXPORT void addProduct_col_unsafe_2(
    int m,
    SciCore::Complex alpha,
    const BlockMatrix& A,
    const BlockDiagonalMatrix& D,
    BlockVector& result);

// Computes A -> D1 A D2
REALTIMETRANSPORT_EXPORT void productCombination_1(
    const BlockDiagonalMatrix& D1,
    BlockMatrix& A,
    const BlockDiagonalMatrix& D2);

// Computes A -> D1 A D2 + D1 A + A D2
REALTIMETRANSPORT_EXPORT void productCombination_2(
    const BlockDiagonalMatrix& D1,
    BlockMatrix& A,
    const BlockDiagonalMatrix& D2);

// Returns α D1 D2 D3
REALTIMETRANSPORT_EXPORT BlockDiagonalMatrix product(
    SciCore::Complex alpha,
    const BlockDiagonalMatrix& D1,
    const BlockDiagonalMatrix& D2,
    const BlockDiagonalMatrix& D3);

// Computes result += alpha * x * D * A
REALTIMETRANSPORT_EXPORT void addProduct(
    SciCore::Complex alpha,
    const Eigen::Matrix<SciCore::Complex, 1, Eigen::Dynamic>& x,
    const BlockDiagonalMatrix& D,
    const BlockMatrix& A,
    Eigen::Matrix<SciCore::Complex, 1, Eigen::Dynamic>& result,
    const std::vector<int>& blockStartIndices);

// Computes result += α \sum_i A1_{ni} A2_{ij} x_j
REALTIMETRANSPORT_EXPORT void addProduct(
    int n,
    SciCore::Complex alpha,
    const BlockMatrix& A1,
    const BlockMatrix& A2,
    const BlockVector& x,
    typename BlockMatrix::MatrixType& result);

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_BLOCK_MATRICES_MATRIX_OPERATIONS_H
