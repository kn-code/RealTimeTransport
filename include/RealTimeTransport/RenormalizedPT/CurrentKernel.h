//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

///
/// \file   CurrentKernel.h
///
/// \brief  Renormalized perturbation theory current kernel.
///

#ifndef REAL_TIME_TRANSPORT_RENORMALIZED_PT_CURRENT_KERNEL_H
#define REAL_TIME_TRANSPORT_RENORMALIZED_PT_CURRENT_KERNEL_H

#include <SciCore/ChebAdaptive.h>

#include "../RealTimeTransport_export.h"
#include "MemoryKernel.h"

namespace RealTimeTransport
{

namespace RenormalizedPT
{

///
/// @ingroup RenPT
///
/// @brief Defines the renormalized current kernel.
///
/// Defines the current kernel computed via the renormalized perturbation theory.
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

    CurrentKernel(const Model* model, int r, Order order, SciCore::Real tMax, SciCore::Real errorGoal, int block = -1)
    {
        _initialize(model, r, order, tMax, errorGoal, nullptr, block);
    }

    CurrentKernel(
        const Model* model,
        int r,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        tf::Executor& executor,
        int block = -1)
    {
        _initialize(model, r, order, tMax, errorGoal, &executor, block);
    }

    ///
    /// @brief Computes the current kernel for a given model.
    ///
    /// Computes the current kernel for a given model.
    ///
    /// @param model        The model for which the current kernel is computed.
    /// @param r            Index of the reservoir \f$ r=0,1,\dots \f$
    /// @param order        The order of the renormalized perturbation series.
    /// @param tMax         Maximum time until the current kernel is resolved.
    /// @param errorGoal    Error goal of the current kernel computation.
    /// @param block        Computes the complete current kernel if \a block==-1, otherwise
    ///                     computes only a single block with index \a block.
    ///
    CurrentKernel(
        const std::unique_ptr<Model>& model,
        int r,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        int block = -1)
    {
        _initialize(model.get(), r, order, tMax, errorGoal, nullptr, block);
    }

    ///
    /// @brief Computes the current kernel for a given model in parallel.
    ///
    /// Computes the current kernel for a given model in parallel.
    ///
    /// @param model        The model for which the current kernel is computed.
    /// @param r            Index of the reservoir \f$ r=0,1,\dots \f$
    /// @param order        The order of the renormalized perturbation series.
    /// @param tMax         Maximum time until the current kernel is resolved.
    /// @param errorGoal    Error goal of the current kernel computation.
    /// @param executor     An excecutor managing multiple threads.
    /// @param block        Computes the complete current kernel if \a block==-1, otherwise
    ///                     computes only a single block with index \a block.
    ///
    CurrentKernel(
        const std::unique_ptr<Model>& model,
        int r,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        tf::Executor& executor,
        int block = -1)
    {
        _initialize(model.get(), r, order, tMax, errorGoal, &executor, block);
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
        const Model* model,
        int r,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        tf::Executor* executor,
        int block);
};

} // namespace RenormalizedPT

///
/// @ingroup RenPT
///
/// @brief Computes the current kernel for a given model.
///
/// Computes the current kernel for a given model.
///
/// @param model        The model for which the current kernel is computed.
/// @param r            Index of the reservoir \f$ r=0,1,\dots \f$
/// @param order        The order of the renormalized perturbation series.
/// @param tMax         Maximum time until the current kernel is resolved.
/// @param errorGoal    Error goal of the current kernel computation.
/// @param block        Computes the complete current kernel if \a block==-1, otherwise
///                     computes only a single block with index \a block.
///
REALTIMETRANSPORT_EXPORT inline RenormalizedPT::CurrentKernel computeCurrentKernel(
    const std::unique_ptr<Model>& model,
    int r,
    RenormalizedPT::Order order,
    SciCore::Real tMax,
    SciCore::Real errorGoal,
    int block = -1)
{
    return RenormalizedPT::CurrentKernel(model, r, order, tMax, errorGoal, block);
}

///
/// @ingroup RenPT
///
/// @brief Computes the current kernel for a given model in parallel.
///
/// Computes the current kernel for a given model in parallel.
///
/// @param model        The model for which the current kernel is computed.
/// @param r            Index of the reservoir \f$ r=0,1,\dots \f$
/// @param order        The order of the renormalized perturbation series.
/// @param tMax         Maximum time until the current kernel is resolved.
/// @param errorGoal    Error goal of the current kernel computation.
/// @param executor     An excecutor managing multiple threads.
/// @param block        Computes the complete current kernel if \a block==-1, otherwise
///                     computes only a single block with index \a block.
///
REALTIMETRANSPORT_EXPORT inline RenormalizedPT::CurrentKernel computeCurrentKernel(
    const std::unique_ptr<Model>& model,
    int r,
    RenormalizedPT::Order order,
    SciCore::Real tMax,
    SciCore::Real errorGoal,
    tf::Executor& executor,
    int block = -1)
{
    return RenormalizedPT::CurrentKernel(model, r, order, tMax, errorGoal, executor, block);
}

///
/// @ingroup RenPT
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
    const RenormalizedPT::CurrentKernel& KCurrent,
    const Propagator& propagator,
    const Model::OperatorType& rho0);

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_RENORMALIZED_PT_CURRENT_KERNEL_H
