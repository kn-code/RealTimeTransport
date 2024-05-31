//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

///
/// \file   MemoryKernel.h
///
/// \brief  Renormalized perturbation theory memory kernel.
///

#ifndef REAL_TIME_TRANSPORT_RENORMALIZED_PT_MEMORY_KERNEL_H
#define REAL_TIME_TRANSPORT_RENORMALIZED_PT_MEMORY_KERNEL_H

#include "../BlockMatrices/BlockDiagonalCheb.h"
#include "../Model.h"
#include "../Propagator.h"
#include "../RealTimeTransport_export.h"

namespace RealTimeTransport
{

namespace RenormalizedPT
{

///
///  \defgroup RenPT 2. Renormalized perturbation theory
///
///  \brief Implementation of the renormalized perturbation theory.
///
///  This page ontains classes and methods related to the implementation of the renormalized perturbation theory.
///
///  \{
///

///
/// @brief Defines the order of the approximation.
///
enum class Order
{
    /// @brief Leading order.
    _1 = 1,

    /// @brief Next-to-leading order.
    _2 = 2
};

///
/// @brief Defines the renormalized memory kernel.
///
/// Defines the memory kernel computed via the renormalized perturbation theory.
///
class REALTIMETRANSPORT_EXPORT MemoryKernel
{
  public:
    /// @brief Constructor.
    MemoryKernel() noexcept;

    /// @brief Move constructor.
    MemoryKernel(MemoryKernel&& other) noexcept;

    /// @brief Copy constructor.
    MemoryKernel(const MemoryKernel& other);

    /// @brief Move assignment operator.
    MemoryKernel& operator=(MemoryKernel&& other);

    /// @brief Copy assignment operator.
    MemoryKernel& operator=(const MemoryKernel& other);

    MemoryKernel(const Model* model, Order order, SciCore::Real tMax, SciCore::Real errorGoal, int block = -1)
    {
        initialize(model, order, tMax, errorGoal, nullptr, block);
    }

    MemoryKernel(
        const Model* model,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        tf::Executor& executor,
        int block = -1)
    {
        initialize(model, order, tMax, errorGoal, &executor, block);
    }

    ///
    /// @brief Computes the memory kernel for a given model.
    ///
    /// Computes the memory kernel for a given model.
    ///
    /// @param model        The model for which the memory kernel is computed.
    /// @param order        The order of the renormalized perturbation series.
    /// @param tMax         Maximum time until the memory kernel is resolved.
    /// @param errorGoal    Error goal of the memory kernel computation.
    /// @param block        Computes the complete memory kernel if \a block==-1, otherwise
    ///                     computes only a single block with index \a block.
    ///
    MemoryKernel(
        const std::unique_ptr<Model>& model,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        int block = -1)
    {
        initialize(model.get(), order, tMax, errorGoal, nullptr, block);
    }

    ///
    /// @brief Computes the memory mernel for a given model in parallel.
    ///
    /// Computes the memory mernel for a given model in parallel.
    ///
    /// @param model        The model for which the memory kernel is computed.
    /// @param order        The order of the renormalized perturbation series.
    /// @param tMax         Maximum time until the memory kernel is resolved.
    /// @param errorGoal    Error goal of the memory kernel computation.
    /// @param executor     An excecutor managing multiple threads.
    /// @param block        Computes the complete memory kernel if \a block==-1, otherwise
    ///                     computes only a single block with index \a block.
    ///
    MemoryKernel(
        const std::unique_ptr<Model>& model,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        tf::Executor& executor,
        int block = -1)
    {
        initialize(model.get(), order, tMax, errorGoal, &executor, block);
    }

    void initialize(
        const Model* model,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        tf::Executor* executor                                      = nullptr,
        int block                                                   = -1,
        SciCore::Real hMin                                          = -1,
        const std::vector<SciCore::RealVector>* initialChebSections = nullptr);

    ///
    /// @brief Operator testing for equality.
    ///
    bool operator==(const MemoryKernel& other) const noexcept;

    ///
    /// @brief Operator testing for inequality.
    ///
    bool operator!=(const MemoryKernel& other) const noexcept;

    ///
    /// @brief Returns a pointer to the model.
    ///
    const Model* model() const noexcept;

    ///
    /// @brief Returns the maximum simulation time.
    ///
    SciCore::Real tMax() const;

    ///
    /// @brief Returns the error goal of the computation.
    ///
    SciCore::Real errorGoal() const noexcept;

    ///
    /// @brief Returns the renormalized Liouvillian \f$-i L_{\infty}\f$.
    ///
    const BlockDiagonalMatrix& LInfty() const noexcept;

    ///
    /// @brief Returns \f$-i K(t)\f$, where \f$K(t)\f$ denotes the renormalized memory kernel.
    ///
    BlockDiagonalCheb& K() noexcept;

    ///
    /// @brief Returns \f$-i K(t)\f$, where \f$K(t)\f$ denotes the renormalized memory kernel.
    ///
    const BlockDiagonalCheb& K() const noexcept;

    ///
    /// @brief Returns \f$-i L_{\infty} -i \hat K(0)\f$, where \f$L_{\infty}\f$ denotes the renormalized Liouvillian and \f$\hat K(0)\f$ the renormalized memory kernel at zero frequency.
    ///
    BlockDiagonalMatrix zeroFrequency() const;

    ///
    /// @brief Returns the stationary state. This method assumes that the stationary state is unique.
    ///
    Model::OperatorType stationaryState(int block = -1) const;

    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(_model, _errorGoal, _minusILInfty, _minusIK);
    }

  private:
    std::unique_ptr<Model> _model;
    SciCore::Real _errorGoal;

    BlockDiagonalMatrix _minusILInfty;
    BlockDiagonalCheb _minusIK;
};

/// \} // end of RenPT

} // namespace RenormalizedPT

///
/// @ingroup RenPT
///
/// @brief Computes the memory kernel for a given model.
///
/// Computes the memory kernel for a given model.
///
/// @param model        The model for which the memory kernel is computed.
/// @param order        The order of the renormalized perturbation series.
/// @param tMax         Maximum time until the memory kernel is resolved.
/// @param errorGoal    Error goal of the memory kernel computation.
/// @param block        Computes the complete memory kernel if \a block==-1, otherwise
///                     computes only a single block with index \a block.
///
REALTIMETRANSPORT_EXPORT inline RenormalizedPT::MemoryKernel computeMemoryKernel(
    const std::unique_ptr<Model>& model,
    RenormalizedPT::Order order,
    SciCore::Real tMax,
    SciCore::Real errorGoal,
    int block = -1)
{
    return RenormalizedPT::MemoryKernel(model, order, tMax, errorGoal, block);
}

///
/// @ingroup RenPT
///
/// @brief Computes the memory mernel for a given model in parallel.
///
/// Computes the memory mernel for a given model in parallel.
///
/// @param model        The model for which the memory kernel is computed.
/// @param order        The order of the renormalized perturbation series.
/// @param tMax         Maximum time until the memory kernel is resolved.
/// @param errorGoal    Error goal of the memory kernel computation.
/// @param executor     An excecutor managing multiple threads.
/// @param block        Computes the complete memory kernel if \a block==-1, otherwise
///                     computes only a single block with index \a block.
///
REALTIMETRANSPORT_EXPORT inline RenormalizedPT::MemoryKernel computeMemoryKernel(
    const std::unique_ptr<Model>& model,
    RenormalizedPT::Order order,
    SciCore::Real tMax,
    SciCore::Real errorGoal,
    tf::Executor& executor,
    int block = -1)
{
    return RenormalizedPT::MemoryKernel(model, order, tMax, errorGoal, executor, block);
}

///
/// @ingroup RenPT
///
/// @brief Computes the propagator corresponding to a given memory kernel.
///
/// Computes the propagator corresponding to a given memory kernel. This is done my numerically
/// solving the time-nonlocal master equation.
///
/// @param memoryKernel     The memory kernel.
/// @param block            Computes the complete propagator if \a block==-1, otherwise
///                         computes only a single block with index \a block.
///
REALTIMETRANSPORT_EXPORT Propagator computePropagator(const RenormalizedPT::MemoryKernel& memoryKernel, int block = -1);

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_RENORMALIZED_PT_MEMORY_KERNEL_H
