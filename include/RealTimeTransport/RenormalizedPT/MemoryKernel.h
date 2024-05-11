//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

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

enum class Order
{
    _1 = 1,
    _2 = 2
};

class REALTIMETRANSPORT_EXPORT MemoryKernel
{
  public:
    MemoryKernel() noexcept;
    MemoryKernel(MemoryKernel&& other) noexcept;
    MemoryKernel(const MemoryKernel& other);
    MemoryKernel& operator=(MemoryKernel&& other);
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

    MemoryKernel(
        const std::unique_ptr<Model>& model,
        Order order,
        SciCore::Real tMax,
        SciCore::Real errorGoal,
        int block = -1)
    {
        initialize(model.get(), order, tMax, errorGoal, nullptr, block);
    }

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
        tf::Executor* executor = nullptr,
        int block              = -1,
        SciCore::Real hMin     = -1,
        const std::vector<SciCore::RealVector>* initialChebSections = nullptr);

    bool operator==(const MemoryKernel& other) const noexcept;
    bool operator!=(const MemoryKernel& other) const noexcept;

    const Model* model() const noexcept;
    SciCore::Real tMax() const;
    SciCore::Real errorGoal() const noexcept;

    ///
    /// @brief Returns -i L_{\infty} -i K(0), where L_{\infty} denotes the renormalized Liouvillian.
    ///
    const Model::BlockDiagonalType& LInfty() const noexcept;

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
    Model::BlockDiagonalType zeroFrequency() const;

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

    Model::BlockDiagonalType _minusILInfty;
    BlockDiagonalCheb _minusIK;
};

} // namespace RenormalizedPT

REALTIMETRANSPORT_EXPORT inline RenormalizedPT::MemoryKernel computeMemoryKernel(
    const std::unique_ptr<Model>& model,
    RenormalizedPT::Order order,
    SciCore::Real tMax,
    SciCore::Real errorGoal,
    int block = -1)
{
    return RenormalizedPT::MemoryKernel(model, order, tMax, errorGoal, block);
}

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

REALTIMETRANSPORT_EXPORT Propagator computePropagator(const RenormalizedPT::MemoryKernel& memoryKernel, int block = -1);

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_RENORMALIZED_PT_MEMORY_KERNEL_H
