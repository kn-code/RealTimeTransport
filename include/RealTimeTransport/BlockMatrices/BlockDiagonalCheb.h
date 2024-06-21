//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

///
/// \file   BlockDiagonalCheb.h
///
/// \brief  Chebyshev interpolation of block diagonal matrices.
///

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

class PortableBinaryInputArchive;
class PortableBinaryOutputArchive;
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

///
/// @brief  This class represents a parameter dependent block diagonal matrix.
///
/// This class represents a parameter dependent block diagonal matrix. The parameter is typically
/// the time \f$ t \f$. The time dependence is represented through a Chebyshev interpolation.
///
class REALTIMETRANSPORT_EXPORT BlockDiagonalCheb
{
  public:
    /// @brief Type representing the matrix elements.
    using Scalar = SciCore::Complex;

    /// @brief Type representing a matrix block at a specific time.
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    ///
    /// @brief Constructs an empty interpolation object.
    ///
    BlockDiagonalCheb();

    ~BlockDiagonalCheb();

    ///
    /// @brief Move constructor.
    ///
    BlockDiagonalCheb(BlockDiagonalCheb&& other) noexcept;

    ///
    /// @brief Copy constructor.
    ///
    BlockDiagonalCheb(const BlockDiagonalCheb& other);

    ///
    /// @brief Move assignment operator.
    ///
    BlockDiagonalCheb& operator=(BlockDiagonalCheb&& other) noexcept;

    ///
    /// @brief Copy assignment operator.
    ///
    BlockDiagonalCheb& operator=(const BlockDiagonalCheb& other);

    ///
    /// \brief              Creates a piecewise Chebyshev interpolation of the function \f$f(t)\f$ with \f$t\in[a,b]\f$.
    ///
    /// \param  f           The function for which the piecewise Chebyshev approximation is computed.
    ///                     The first parameter refers to the block index, the second to the time argument.
    /// \param  nBlocks     The number of blocks.
    /// \param  a           Lower interval point.
    /// \param  b           Upper interval point.
    /// \param  epsAbs      Absolute error goal.
    /// \param  epsRel      Relative error goal.
    /// \param  hMin        The minimum allowed interval length.
    /// \param  ok          Set to _true_ if error goal was achieved, otherwise set to _false_.
    ///
    BlockDiagonalCheb(
        const std::function<MatrixType(int, SciCore::Real)>& f,
        int nBlocks,
        SciCore::Real a,
        SciCore::Real b,
        SciCore::Real epsAbs,
        SciCore::Real epsRel,
        SciCore::Real hMin,
        bool* ok = nullptr);

    ///
    /// \brief  Creates a piecewise Chebyshev interpolation of the function \f$f(t)\f$ with \f$t\in[a,b]\f$ in parallel.
    ///
    /// \param  f           The function for which the piecewise Chebyshev approximation is computed.
    ///                     The first parameter refers to the block index, the second to the time argument.
    /// \param  nBlocks     The number of blocks.
    /// \param  a           Lower interval point.
    /// \param  b           Upper interval point.
    /// \param  epsAbs      Absolute error goal.
    /// \param  epsRel      Relative error goal.
    /// \param  hMin        The minimum allowed interval length.
    /// \param  executor    Taskflow executor managing the threads.
    /// \param  ok          Set to _true_ if error goal was achieved, otherwise set to _false_.
    ///
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

    ///
    /// @brief Equality comparison operator.
    ///
    bool operator==(const BlockDiagonalCheb& other) const;

    ///
    /// @brief Inequality comparison operator.
    ///
    bool operator!=(const BlockDiagonalCheb& other) const;

    ///
    /// @brief Returns the number of matrix blocks.
    ///
    int numBlocks() const noexcept;

    ///
    /// @brief Lower limit of the interplation interval.
    ///
    SciCore::Real lowerLimit() const;

    ///
    /// @brief Upper limit of the interplation interval.
    ///
    SciCore::Real upperLimit() const;

    ///
    /// @brief Returns the full interpolated matrix at time \f$ t \f$.
    ///
    BlockDiagonalMatrix operator()(SciCore::Real t) const;

    ///
    /// @brief Returns block \a i of the interpolated matrix at time \f$ t \f$.
    ///
    MatrixType operator()(int i, SciCore::Real t) const;

    ///
    /// @brief Returns the interpolation object for block \a i.
    ///
    const SciCore::ChebAdaptive<MatrixType>& block(int i) const;

    ///
    /// \brief Computes the derivative of the represented function and returns it as a new object.
    ///
    BlockDiagonalCheb diff() const;

    ///
    /// \brief Computes the integral of the represented function and returns it as a new object.
    ///
    BlockDiagonalCheb integrate() const;

    std::vector<SciCore::RealVector> sections() const;

    void serialize(cereal::BinaryInputArchive& archive);
    void serialize(cereal::BinaryOutputArchive& archive);
    void serialize(cereal::PortableBinaryInputArchive& archive);
    void serialize(cereal::PortableBinaryOutputArchive& archive);

  private:
    struct Impl;
    std::unique_ptr<Impl> _pimpl;
};

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_DIAGONAL_CHEB_H
