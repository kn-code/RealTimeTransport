//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

///
/// \file   ConductanceKernel.h
///
/// \brief  Iterated RG conductance kernel.
///

#ifndef REAL_TIME_TRANSPORT_ITERATED_RG_CONDUCTANCE_KERNEL_H
#define REAL_TIME_TRANSPORT_ITERATED_RG_CONDUCTANCE_KERNEL_H

#include "../RealTimeTransport_export.h"
#include "CurrentKernel.h"

namespace RealTimeTransport
{

namespace IteratedRG
{

///
/// @ingroup IterRG
///
/// @brief Defines the conductance kernel using the self consistent RG method.
///
/// This class computes the conductance kernel using the self consistent RG memory kernel. The class can be used
/// to compute the stationary conductance \f$ dI_r/d\mu_r \f$ of reservoir \f$ r \f$ without numerical differentiation,
/// which is numerically more stable. The memory and current kernel must be computed first in order to use this class.
/// A cross conductance \f$ dI_r/d\mu_{r'} \f$ can be obtained by first computing the current kernel for resrevoir
/// \f$ r \f$, and then the conductance kernel for reservoir \f$ r' \f$.
///
class REALTIMETRANSPORT_EXPORT ConductanceKernel
{
  public:
    /// @brief Constructor.
    ConductanceKernel() noexcept;

    /// @brief Move constructor.
    ConductanceKernel(ConductanceKernel&& other) noexcept;

    /// @brief Copy constructor.
    ConductanceKernel(const ConductanceKernel& other);

    /// @brief Move assignment operator.
    ConductanceKernel& operator=(ConductanceKernel&& other);

    /// @brief Copy assignment operator.
    ConductanceKernel& operator=(const ConductanceKernel& other);

    ///
    /// @brief Computes the conductance kernel for a given model.
    ///
    /// Computes the conductance kernel for a given model.
    ///
    /// @param K                The memory kernel of the model.
    /// @param KI               The current kernel of the model.
    /// @param Pi               Propagator of the model.
    /// @param stationaryState  The Stationary state  of the model.
    /// @param r                Index of the reservoir \f$ r=0,1,\dots \f$
    /// @param order            The order of the approximation.
    /// @param tMax             Maximum time until the conductance kernel is resolved.
    /// @param errorGoal        Error goal of the conductance kernel computation.
    /// @param block            Computes the complete conductance kernel if \a block==-1, otherwise
    ///                         computes only a single block with index \a block.
    ///
    ConductanceKernel(
        const MemoryKernel& K,
        const CurrentKernel& KI,
        const Propagator& Pi,
        const Model::OperatorType& stationaryState,
        int r,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        int block = -1)
    {
        _initialize(K, KI, Pi, stationaryState, r, order, tMax, errorGoal, nullptr, block);
    }

    ///
    /// @brief Computes the conductance kernel for a given model in parallel.
    ///
    /// Computes the conductance kernel for a given model in parallel.
    ///
    /// @param K                The memory kernel of the model.
    /// @param KI               The current kernel of the model.
    /// @param Pi               Propagator of the model.
    /// @param stationaryState  The Stationary state  of the model.
    /// @param r                Index of the reservoir \f$ r=0,1,\dots \f$
    /// @param order            The order of the approximation.
    /// @param tMax             Maximum time until the conductance kernel is resolved.
    /// @param errorGoal        Error goal of the conductance kernel computation.
    /// @param executor         A Taskflow executor managing threads.
    /// @param block            Computes the complete conductance kernel if \a block==-1, otherwise
    ///                         computes only a single block with index \a block.
    ///
    ConductanceKernel(
        const MemoryKernel& K,
        const CurrentKernel& KI,
        const Propagator& Pi,
        const Model::OperatorType& stationaryState,
        int r,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        tf::Executor& executor,
        int block = -1)
    {
        _initialize(K, KI, Pi, stationaryState, r, order, tMax, errorGoal, &executor, block);
    }

    ///
    /// @brief Returns the reservoir index for which the conductance kernel was computed.
    ///
    int r() const noexcept;

    ///
    /// @brief Returns the maximum simulation time.
    ///
    SciCore::Real tMax() const;

    ///
    /// @brief Returns the error goal of the computation.
    ///
    SciCore::Real errorGoal() const noexcept;

    ///
    /// @brief Returns the (stationary) conductance.
    ///
    SciCore::Real conductance() const;

    ///
    /// @brief Returns the stationary state derivative \f$ d\rho/d\mu_r \f$.
    ///
    Model::OperatorType dState() const;

    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(
            _model, _r, _errorGoal, _rhoStat, _d_dmu_rhoStat, _currentKernelZeroFreq, _d_dmu_memoryKernel,
            _d_dmu_currentKernel);
    }

  private:
    std::unique_ptr<Model> _model;
    int _r;
    SciCore::Real _errorGoal;

    Model::SupervectorType _rhoStat;
    Model::SupervectorType _d_dmu_rhoStat;
    Model::SuperRowVectorType _currentKernelZeroFreq;
    BlockDiagonalCheb _d_dmu_memoryKernel;
    SciCore::ChebAdaptive<Model::SuperRowVectorType> _d_dmu_currentKernel;

    void _initialize(
        const MemoryKernel& K,
        const CurrentKernel& KI,
        const Propagator& Pi,
        const Model::OperatorType& stationaryState,
        int r,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        tf::Executor* executor,
        int block,
        SciCore::Real hMin = -1);
};

} // namespace IteratedRG

///
/// @ingroup IterRG
///
/// @brief Computes the conductance kernel for a given model in the same reservoir for which the current kernel was computed.
///
/// Computes the conductance kernel for a given model in the same reservoir for which the current kernel was computed.
///
/// @param K        The memory kernel for a given model.
/// @param KI       The current kernel for a given model.
/// @param Pi       Propagator of the model.
/// @param order    The order of the approximation.
/// @param block    Computes the complete conductance kernel if \a block==-1, otherwise
///                 computes only a single block with index \a block.
///
REALTIMETRANSPORT_EXPORT inline IteratedRG::ConductanceKernel computeConductanceKernel(
    const IteratedRG::MemoryKernel& K,
    const IteratedRG::CurrentKernel& KI,
    const Propagator& Pi,
    IteratedRG::Order order,
    int block = -1)
{
    return IteratedRG::ConductanceKernel(
        K, KI, Pi, K.stationaryState(block), KI.r(), order, K.tMax(), K.errorGoal(), block);
}

///
/// @ingroup IterRG
///
/// @brief Computes the conductance kernel for a given model in the same reservoir for which the current kernel was computed in parallel.
///
/// Computes the conductance kernel for a given model in the same reservoir for which the current kernel was computed in parallel.
///
/// @param K        The memory kernel for a given model.
/// @param KI       The current kernel for a given model.
/// @param Pi       Propagator of the model.
/// @param order    The order of the approximation.
/// @param executor An excecutor managing multiple threads.
/// @param block    Computes the complete conductance kernel if \a block==-1, otherwise
///                 computes only a single block with index \a block.
///
REALTIMETRANSPORT_EXPORT inline IteratedRG::ConductanceKernel computeConductanceKernel(
    const IteratedRG::MemoryKernel& K,
    const IteratedRG::CurrentKernel& KI,
    const Propagator& Pi,
    IteratedRG::Order order,
    tf::Executor& executor,
    int block = -1)
{
    return IteratedRG::ConductanceKernel(
        K, KI, Pi, K.stationaryState(block), KI.r(), order, K.tMax(), K.errorGoal(), executor, block);
}

///
/// @ingroup IterRG
///
/// @brief Computes the conductance kernel for a given model.
///
/// Computes the conductance kernel which gives access to the (cross)conductance \f$dI_{r_1}/d\mu_{r_2}\f$ for a
/// given model. Here \f$r_1\f$ is determined by the current kernel _KI_, and \f$r_2\f$ by the parameter _r_.
///
/// @param K        The memory kernel for a given model.
/// @param KI       The current kernel for a given model.
/// @param Pi       Propagator of the model.
/// @param r        Reservoir at which the derivative of the chemical potential is taken.
/// @param order    The order of the approximation.
/// @param block    Computes the complete conductance kernel if \a block==-1, otherwise
///                 computes only a single block with index \a block.
///
REALTIMETRANSPORT_EXPORT inline IteratedRG::ConductanceKernel computeConductanceKernel(
    const IteratedRG::MemoryKernel& K,
    const IteratedRG::CurrentKernel& KI,
    const Propagator& Pi,
    int r,
    IteratedRG::Order order,
    int block = -1)
{
    return IteratedRG::ConductanceKernel(K, KI, Pi, K.stationaryState(block), r, order, K.tMax(), K.errorGoal(), block);
}

///
/// @ingroup IterRG
///
/// @brief Computes the conductance kernel for a given model in parallel.
///
/// Computes the conductance kernel which gives access to the (cross)conductance \f$dI_{r_1}/d\mu_{r_2}\f$ for a given
/// model in parallel. Here \f$r_1\f$ is determined by the current kernel _KI_, and \f$r_2\f$ by the parameter _r_.
///
/// @param K        The memory kernel for a given model.
/// @param KI       The current kernel for a given model.
/// @param Pi       Propagator of the model.
/// @param r        Reservoir at which the derivative of the chemical potential is taken.
/// @param order    The order of the approximation.
/// @param executor An excecutor managing multiple threads.
/// @param block    Computes the complete conductance kernel if \a block==-1, otherwise
///                 computes only a single block with index \a block.
///
REALTIMETRANSPORT_EXPORT inline IteratedRG::ConductanceKernel computeConductanceKernel(
    const IteratedRG::MemoryKernel& K,
    const IteratedRG::CurrentKernel& KI,
    const Propagator& Pi,
    int r,
    IteratedRG::Order order,
    tf::Executor& executor,
    int block = -1)
{
    return IteratedRG::ConductanceKernel(
        K, KI, Pi, K.stationaryState(block), r, order, K.tMax(), K.errorGoal(), executor, block);
}

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_ITERATED_RG_CONDUCTANCE_KERNEL_H
