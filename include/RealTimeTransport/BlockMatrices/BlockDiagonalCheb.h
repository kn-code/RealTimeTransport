//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_DIAGONAL_CHEB_H
#define REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_DIAGONAL_CHEB_H

#include <functional>
#include <memory>

#include <SciCore/Definitions.h>

#include "../RealTimeTransport_export.h"
#include "BlockDiagonalMatrix.h"

namespace cereal
{
class BinaryInputArchive;
class BinaryOutputArchive;
} // namespace cereal

namespace SciCore
{
template <MatrixOrScalarType T>
class ChebAdaptive;
}

namespace tf
{
class Executor;
}

namespace RealTimeTransport
{

class REALTIMETRANSPORT_EXPORT BlockDiagonalCheb
{
  public:
    using Scalar     = SciCore::Complex;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    BlockDiagonalCheb();
    ~BlockDiagonalCheb();
    BlockDiagonalCheb(BlockDiagonalCheb&& other) noexcept;
    BlockDiagonalCheb(const BlockDiagonalCheb& other);
    BlockDiagonalCheb& operator=(BlockDiagonalCheb&& other) noexcept;
    BlockDiagonalCheb& operator=(const BlockDiagonalCheb& other);

    // f(i, t) returns block i at time t
    BlockDiagonalCheb(
        const std::function<MatrixType(int, SciCore::Real)>& f,
        int nBlocks,
        SciCore::Real a,
        SciCore::Real b,
        SciCore::Real epsAbs,
        SciCore::Real epsRel,
        SciCore::Real hMin,
        bool* ok = nullptr);

    // f(i, t) return block i at time t
    BlockDiagonalCheb(
        const std::function<MatrixType(int, SciCore::Real)>& f,
        int nBlocks,
        SciCore::Real a,
        SciCore::Real b,
        SciCore::Real epsAbs,
        SciCore::Real epsRel,
        SciCore::Real hMin,
        tf::Executor& executor,
        bool* ok = nullptr);

    // f(i, t) returns block i at time t
    BlockDiagonalCheb(
        const std::function<MatrixType(int, SciCore::Real)>& f,
        int nBlocks,
        const std::vector<SciCore::RealVector>& sections,
        SciCore::Real epsAbs,
        SciCore::Real epsRel,
        SciCore::Real hMin,
        bool* ok = nullptr);

    BlockDiagonalCheb(
        const std::function<MatrixType(int, SciCore::Real)>& f,
        int nBlocks,
        const std::vector<SciCore::RealVector>& sections,
        SciCore::Real epsAbs,
        SciCore::Real epsRel,
        SciCore::Real hMin,
        tf::Executor& executor,
        bool* ok = nullptr);

    BlockDiagonalCheb(std::vector<SciCore::ChebAdaptive<MatrixType>>&& blocks) noexcept;

    bool operator==(const BlockDiagonalCheb& other) const;
    bool operator!=(const BlockDiagonalCheb& other) const;

    int numBlocks() const noexcept;

    SciCore::Real lowerLimit() const;
    SciCore::Real upperLimit() const;

    BlockDiagonalMatrix operator()(SciCore::Real t) const;
    MatrixType operator()(int i, SciCore::Real t) const;

    const SciCore::ChebAdaptive<MatrixType>& block(int i) const;

    BlockDiagonalCheb diff() const;
    BlockDiagonalCheb integrate() const;

    std::vector<SciCore::RealVector> sections() const;

    void serialize(cereal::BinaryInputArchive& archive);
    void serialize(cereal::BinaryOutputArchive& archive);

  private:
    struct Impl;
    std::unique_ptr<Impl> _pimpl;
};

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_DIAGONAL_CHEB_H