//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

///
/// \file   MemoryKernel.h
///
/// \brief  Iterated renormalization group memory kernel.
///

#ifndef REAL_TIME_TRANSPORT_ITERATED_RG_MEMORY_KERNEL_H
#define REAL_TIME_TRANSPORT_ITERATED_RG_MEMORY_KERNEL_H

#include "../BlockMatrices/BlockDiagonalCheb.h"
#include "../BlockMatrices/BlockDiagonalMatrix.h"
#include "../BlockMatrices/BlockVector.h"
#include "../Model.h"
#include "../Propagator.h"
#include "../RealTimeTransport_export.h"

namespace RealTimeTransport
{

namespace IteratedRG
{

///
///  \defgroup IterRG 3. Renormalization group methods
///
///  \brief Implementation of the renormalization group iteration.
///
///  This page ontains classes and methods related to the implementation of the renormalization group iteration.
///
///  \{
///

///
/// @brief Defines the order of the approximation.
///
enum class Order
{
    /// @brief Two-loop approximation.
    _2 = 2,

    /// @brief Three-loop approximation.
    _3 = 3
};

///
/// @brief Defines the RG memory kernel.
///
/// Defines the memory kernel computed via the renormalization group iteration.
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

    MemoryKernel(const Model* model, Order order, SciCore::Real tMax, SciCore::Real errorGoal)
    {
        initialize(model, order, tMax, errorGoal, nullptr);
    }

    MemoryKernel(const Model* model, Order order, SciCore::Real tMax, SciCore::Real errorGoal, tf::Executor& executor)
    {
        initialize(model, order, tMax, errorGoal, &executor);
    }

    ///
    /// @brief Computes the memory kernel for a given model.
    ///
    /// Computes the memory kernel for a given model using the renormalization group iteration.
    ///
    /// @param model        The model for which the memory kernel is computed.
    /// @param order        Order of the RG iteration.
    /// @param tMax         Maximum time until the memory kernel is resolved.
    /// @param errorGoal    Error goal of the memory kernel computation.
    ///
    MemoryKernel(const std::unique_ptr<Model>& model, Order order, SciCore::Real tMax, SciCore::Real errorGoal)
    {
        initialize(model.get(), order, tMax, errorGoal, nullptr);
    }

    ///
    /// @brief Computes the memory mernel for a given model in parallel.
    ///
    /// Computes the memory kernel for a given model using the renormalization group iteration in parallel.
    ///
    /// @param model        The model for which the memory kernel is computed.
    /// @param order        Order of the RG iteration.
    /// @param tMax         Maximum time until the memory kernel is resolved.
    /// @param errorGoal    Error goal of the memory kernel computation.
    /// @param executor     An excecutor managing multiple threads.
    ///
    MemoryKernel(
        const std::unique_ptr<Model>& model,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        tf::Executor& executor)
    {
        initialize(model.get(), order, tMax, errorGoal, &executor);
    }

    void initialize(
        const Model* model,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        tf::Executor* executor,
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
    /// @brief Returns \f$-i K(t)\f$, where \f$K(t)\f$ denotes the memory kernel.
    ///
    BlockDiagonalCheb& K() noexcept;

    ///
    /// @brief Returns \f$-i K(t)\f$, where \f$K(t)\f$ denotes the memory kernel.
    ///
    const BlockDiagonalCheb& K() const noexcept;

    ///
    /// @brief Returns \f$-i L_{\infty} -i \hat K(0)\f$, where \f$L_{\infty}\f$ denotes the renormalized Liouvillian and \f$\hat K(0)\f$ the memory kernel at zero frequency.
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

    // Initializes _minusIK with perturbation theory and returns corresponding propagator
    BlockDiagonalCheb _initMemoryKernel(
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        tf::Executor* executor,
        SciCore::Real hMin,
        const std::vector<SciCore::RealVector>* initialChebSections);

    // Returns the effective vertex (without delta singular part) of order D^3 + D^5
    BlockVector _computeD_O3_O5_col(
        int i,
        int col,
        SciCore::Real t_minus_tau,
        SciCore::Real tau_minus_s,
        SciCore::Real epsAbs,
        const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
        const std::function<BlockVector(int, int, SciCore::Real, SciCore::Real)>& computeD_O3_col,
        const std::vector<Model::SuperfermionType>& superfermion);
};

/// \} // end of IterRG

} // namespace IteratedRG

///
/// @ingroup IterRG
///
/// @brief Computes the memory kernel for a given model.
///
/// Computes the memory kernel for a given model using the renormalization group iteration.
///
/// @param model        The model for which the memory kernel is computed.
/// @param order        Order of the RG iteration.
/// @param tMax         Maximum time until the memory kernel is resolved.
/// @param errorGoal    Error goal of the memory kernel computation.
///
REALTIMETRANSPORT_EXPORT inline IteratedRG::MemoryKernel computeMemoryKernel(
    const std::unique_ptr<Model>& model,
    IteratedRG::Order order,
    SciCore::Real tMax,
    SciCore::Real errorGoal)
{
    return IteratedRG::MemoryKernel(model, order, tMax, errorGoal);
}

///
/// @ingroup IterRG
///
/// @brief Computes the memory mernel for a given model in parallel.
///
/// Computes the memory kernel for a given model using the renormalization group iteration in parallel.
///
/// @param model        The model for which the memory kernel is computed.
/// @param order        Order of the RG iteration.
/// @param tMax         Maximum time until the memory kernel is resolved.
/// @param errorGoal    Error goal of the memory kernel computation.
/// @param executor     An excecutor managing multiple threads.
///
REALTIMETRANSPORT_EXPORT inline IteratedRG::MemoryKernel computeMemoryKernel(
    const std::unique_ptr<Model>& model,
    IteratedRG::Order order,
    SciCore::Real tMax,
    SciCore::Real errorGoal,
    tf::Executor& executor)
{
    return IteratedRG::MemoryKernel(model, order, tMax, errorGoal, executor);
}

///
/// @ingroup IterRG
///
/// @brief Computes the propagator corresponding to a given memory kernel.
///
/// Computes the propagator corresponding to a given memory kernel. This is done my numerically
/// solving the time-nonlocal master equation.
///
/// @param memoryKernel     The memory kernel.
///
REALTIMETRANSPORT_EXPORT Propagator computePropagator(const IteratedRG::MemoryKernel& memoryKernel);

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_ITERATED_RG_MEMORY_KERNEL_H
