//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

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

enum class Order
{
    _2 = 2,
    _3 = 3
};

class REALTIMETRANSPORT_EXPORT MemoryKernel
{
  public:
    MemoryKernel() noexcept;
    MemoryKernel(MemoryKernel&& other) noexcept;
    MemoryKernel(const MemoryKernel& other);
    MemoryKernel& operator=(MemoryKernel&& other);
    MemoryKernel& operator=(const MemoryKernel& other);

    MemoryKernel(const Model* model, Order order, SciCore::Real tMax, SciCore::Real errorGoal)
    {
        initialize(model, order, tMax, errorGoal, nullptr);
    }

    MemoryKernel(const Model* model, Order order, SciCore::Real tMax, SciCore::Real errorGoal, tf::Executor& executor)
    {
        initialize(model, order, tMax, errorGoal, &executor);
    }

    MemoryKernel(const std::unique_ptr<Model>& model, Order order, SciCore::Real tMax, SciCore::Real errorGoal)
    {
        initialize(model.get(), order, tMax, errorGoal, nullptr);
    }

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

    bool operator==(const MemoryKernel& other) const noexcept;
    bool operator!=(const MemoryKernel& other) const noexcept;

    const Model* model() const noexcept;
    SciCore::Real tMax() const;
    SciCore::Real errorGoal() const noexcept;

    ///
    /// @brief Returns -i L_{\infty} -i K(0), where L_{\infty} denotes the renormalized Liouvillian.
    ///
    const BlockDiagonalMatrix& LInfty() const noexcept;

    ///
    /// @brief Returns -i K, where K denotes the memory kernel.
    ///
    BlockDiagonalCheb& K() noexcept;

    ///
    /// @brief Returns -i K, where K denotes the memory kernel.
    ///
    const BlockDiagonalCheb& K() const noexcept;

    ///
    /// @brief Returns -i L_{\infty} -i K(0), where L_{\infty} denotes the renormalized Liouvillian and K(0) the memory kernel at zero frequency.
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

} // namespace IteratedRG

REALTIMETRANSPORT_EXPORT inline IteratedRG::MemoryKernel computeMemoryKernel(
    const std::unique_ptr<Model>& model,
    IteratedRG::Order order,
    SciCore::Real tMax,
    SciCore::Real errorGoal)
{
    return IteratedRG::MemoryKernel(model, order, tMax, errorGoal);
}

REALTIMETRANSPORT_EXPORT inline IteratedRG::MemoryKernel computeMemoryKernel(
    const std::unique_ptr<Model>& model,
    IteratedRG::Order order,
    SciCore::Real tMax,
    SciCore::Real errorGoal,
    tf::Executor& executor)
{
    return IteratedRG::MemoryKernel(model, order, tMax, errorGoal, executor);
}

REALTIMETRANSPORT_EXPORT Propagator computePropagator(const IteratedRG::MemoryKernel& memoryKernel);

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_ITERATED_RG_MEMORY_KERNEL_H
