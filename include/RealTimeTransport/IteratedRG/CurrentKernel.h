//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_ITERATED_RG_CURRENT_KERNEL_H
#define REAL_TIME_TRANSPORT_ITERATED_RG_CURRENT_KERNEL_H

#include <SciCore/ChebAdaptive.h>

#include "../RealTimeTransport_export.h"
#include "MemoryKernel.h"

namespace RealTimeTransport
{

namespace IteratedRG
{

///
/// @ingroup IterRG
///
/// @brief Defines the current kernel.
///
/// Defines the current kernel computed via the renormalization group iteration.
///
class REALTIMETRANSPORT_EXPORT CurrentKernel
{
  public:
    /// @brief Constructor.
    CurrentKernel() noexcept;

    /// @brief Move constructor.
    CurrentKernel(CurrentKernel&& other) noexcept;

    /// @brief Copy constructor.
    CurrentKernel(const CurrentKernel& other);

    /// @brief Move assignment operator.
    CurrentKernel& operator=(CurrentKernel&& other);

    /// @brief Copy assignment operator.
    CurrentKernel& operator=(const CurrentKernel& other);

    ///
    /// @brief Computes the current kernel for a given model.
    ///
    /// Computes the current kernel for a given model using the renormalization group iteration.
    ///
    /// @param K            The memory kernel for the model.
    /// @param propagator   The propagator for the model.
    /// @param r            Index of the reservoir \f$ r=0,1,\dots \f$
    /// @param order        The loop-order of the renormalization group iteration.
    /// @param tMax         Maximum time until the current kernel is resolved.
    /// @param errorGoal    Error goal of the current kernel computation.
    /// @param block        Computes the complete current kernel if \a block==-1, otherwise
    ///                     computes only a single block with index \a block.
    ///
    CurrentKernel(
        const MemoryKernel& K,
        const Propagator& propagator,
        int r,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        int block = -1)
    {
        _initialize(K, propagator, r, order, tMax, errorGoal, nullptr, block);
    }

    ///
    /// @brief Computes the current kernel for a given model in parallel.
    ///
    /// Computes the current kernel for a given model using the renormalization group iteration in parallel.
    ///
    /// @param K            The memory kernel for the model.
    /// @param propagator   The propagator for the model.
    /// @param r            Index of the reservoir \f$ r=0,1,\dots \f$
    /// @param order        The loop-order of the renormalization group iteration.
    /// @param tMax         Maximum time until the current kernel is resolved.
    /// @param errorGoal    Error goal of the current kernel computation.
    /// @param executor     An excecutor managing multiple threads.
    /// @param block        Computes the complete current kernel if \a block==-1, otherwise
    ///                     computes only a single block with index \a block.
    ///
    CurrentKernel(
        const MemoryKernel& K,
        const Propagator& propagator,
        int r,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        tf::Executor& executor,
        int block = -1)
    {
        _initialize(K, propagator, r, order, tMax, errorGoal, &executor, block);
    }

    ///
    /// @brief Returns a pointer to the model.
    ///
    const Model* model() const noexcept;

    ///
    /// @brief Returns the reservoir index for which the current kernel was computed.
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
    /// @brief Returns \f$ -i \Sigma_{\infty} \f$, where \f$ \Sigma_{\infty} \f$ denotes the infinite temperature current kernel.
    ///
    const Model::SuperRowVectorType& SigmaInfty() const noexcept;

    ///
    /// @brief Returns \f$ -i K \f$, where \f$ K \f$ denotes the current kernel.
    ///
    const SciCore::ChebAdaptive<Model::SuperRowVectorType>& K() const noexcept;

    ///
    /// @brief Returns \f$-i \Sigma_{\infty} -i K(0)\f$, where \f$\Sigma_{\infty}\f$ denotes infinite temperature current kernel and \f$K(0)\f$ the current kernel at zero frequency.
    ///
    Model::SuperRowVectorType zeroFrequency() const;

    ///
    /// @brief Returns the stationary current.
    ///
    SciCore::Real stationaryCurrent(const Model::OperatorType& stationaryState) const;

    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(_model, _r, _errorGoal, _minusISigmaInfty, _minusIK);
    }

  private:
    std::unique_ptr<Model> _model;
    int _r;
    SciCore::Real _errorGoal;

    Model::SuperRowVectorType _minusISigmaInfty;
    SciCore::ChebAdaptive<Model::SuperRowVectorType> _minusIK;

    void _initialize(
        const MemoryKernel& K,
        const Propagator& propagator,
        int r,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        tf::Executor* executor,
        int block);
};

} // namespace IteratedRG

///
/// @ingroup IterRG
///
/// @brief Computes the current kernel for a given model.
///
/// Computes the current kernel for a given model using the renormalization group iteration.
///
/// @param K            The memory kernel for the model.
/// @param propagator   The propagator for the model.
/// @param r            Index of the reservoir \f$ r=0,1,\dots \f$
/// @param order        The loop-order of the renormalization group iteration.
/// @param tMax         Maximum time until the current kernel is resolved.
/// @param errorGoal    Error goal of the current kernel computation.
/// @param block        Computes the complete current kernel if \a block==-1, otherwise
///                     computes only a single block with index \a block.
///
REALTIMETRANSPORT_EXPORT inline IteratedRG::CurrentKernel computeCurrentKernel(
    const IteratedRG::MemoryKernel& K,
    const Propagator& propagator,
    int r,
    IteratedRG::Order order,
    SciCore::Real tMax,
    SciCore::Real errorGoal,
    int block = -1)
{
    return IteratedRG::CurrentKernel(K, propagator, r, order, tMax, errorGoal, block);
}

///
/// @ingroup IterRG
///
/// @brief Computes the current kernel for a given model in parallel.
///
/// Computes the current kernel for a given model using the renormalization group iteration in parallel.
///
/// @param K            The memory kernel for the model.
/// @param propagator   The propagator for the model.
/// @param r            Index of the reservoir \f$ r=0,1,\dots \f$
/// @param order        The loop-order of the renormalization group iteration.
/// @param tMax         Maximum time until the current kernel is resolved.
/// @param errorGoal    Error goal of the current kernel computation.
/// @param executor     An excecutor managing multiple threads.
/// @param block        Computes the complete current kernel if \a block==-1, otherwise
///                     computes only a single block with index \a block.
///
REALTIMETRANSPORT_EXPORT inline IteratedRG::CurrentKernel computeCurrentKernel(
    const IteratedRG::MemoryKernel& K,
    const Propagator& propagator,
    int r,
    IteratedRG::Order order,
    SciCore::Real tMax,
    SciCore::Real errorGoal,
    tf::Executor& executor,
    int block = -1)
{
    return IteratedRG::CurrentKernel(K, propagator, r, order, tMax, errorGoal, executor, block);
}

///
/// @ingroup IterRG
///
/// @brief Computes the transient current for a given initial state.
///
/// Computes the transient current for a given initial state.
///
/// @param KCurrent     The current kernel.
/// @param propagator   The propagator of the dynamics.
/// @param rho0         The initial state.
///
REALTIMETRANSPORT_EXPORT SciCore::ChebAdaptive<SciCore::Real> computeCurrent(
    const IteratedRG::CurrentKernel& KCurrent,
    const Propagator& propagator,
    const Model::OperatorType& rho0);

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_ITERATED_RG_CURRENT_KERNEL_H
