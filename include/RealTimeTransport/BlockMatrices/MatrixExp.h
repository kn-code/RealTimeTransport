//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_BLOCK_MATRICES_MATRIX_EXP_H
#define REAL_TIME_TRANSPORT_BLOCK_MATRICES_MATRIX_EXP_H

#include <vector>

#include <SciCore/Definitions.h>

#include "../Error.h"
#include "../RealTimeTransport_export.h"
#include "BlockDiagonalMatrix.h"

namespace RealTimeTransport
{

// Computes for a given Matrix A the matrix exponential exp(t*A)
class REALTIMETRANSPORT_EXPORT MatrixExp
{
  public:
    MatrixExp() noexcept                                  = default;
    MatrixExp(MatrixExp&& other) noexcept                 = default;
    MatrixExp(const MatrixExp& other)                     = default;
    MatrixExp& operator=(MatrixExp&& other) noexcept      = default;
    MatrixExp& operator=(const MatrixExp& other) noexcept = default;

    MatrixExp(const SciCore::Matrix& A);

    void initialize(const SciCore::Matrix& A);

    SciCore::Matrix operator()(SciCore::Real t) const;
    SciCore::Matrix expm1(SciCore::Real t) const;

  private:
    bool _useEigendecomposition;
    SciCore::Vector _eigenvalues;
    std::vector<SciCore::Matrix> _projectors;
};

class REALTIMETRANSPORT_EXPORT BlockDiagonalMatrixExp
{
  public:
    BlockDiagonalMatrixExp() noexcept                                               = default;
    BlockDiagonalMatrixExp(BlockDiagonalMatrixExp&& other) noexcept                 = default;
    BlockDiagonalMatrixExp(const BlockDiagonalMatrixExp& other)                     = default;
    BlockDiagonalMatrixExp& operator=(BlockDiagonalMatrixExp&& other) noexcept      = default;
    BlockDiagonalMatrixExp& operator=(const BlockDiagonalMatrixExp& other) noexcept = default;

    BlockDiagonalMatrixExp(const BlockDiagonalMatrix& A);

    void initialize(const BlockDiagonalMatrix& A);

    BlockDiagonalMatrix operator()(SciCore::Real t) const;
    BlockDiagonalMatrix::MatrixType operator()(int blockIndex, SciCore::Real t) const;

    BlockDiagonalMatrix expm1(SciCore::Real t) const;
    BlockDiagonalMatrix::MatrixType expm1(int blockIndex, SciCore::Real t) const;

  private:
    std::vector<MatrixExp> _blocks;
};

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_BLOCK_MATRICES_MATRIX_EXP_H
