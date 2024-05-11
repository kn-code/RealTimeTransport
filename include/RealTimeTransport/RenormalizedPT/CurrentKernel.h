//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_RENORMALIZED_PT_CURRENT_KERNEL_H
#define REAL_TIME_TRANSPORT_RENORMALIZED_PT_CURRENT_KERNEL_H

#include <SciCore/ChebAdaptive.h>

#include "MemoryKernel.h"
#include "../RealTimeTransport_export.h"

namespace RealTimeTransport
{

namespace RenormalizedPT
{

class REALTIMETRANSPORT_EXPORT CurrentKernel
{
  public:
    CurrentKernel() noexcept                             = default;
    CurrentKernel(CurrentKernel&& other) noexcept        = default;
    CurrentKernel(const CurrentKernel& other)            = default;
    CurrentKernel& operator=(CurrentKernel&& other)      = default;
    CurrentKernel& operator=(const CurrentKernel& other) = default;

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

    const Model* model() const noexcept;
    SciCore::Real tMax() const;
    SciCore::Real errorGoal() const noexcept;

    ///
    /// @brief Returns -i \Sigma_{\infty}, where \Sigma_{\infty} denotes the infinite temperature current kernel.
    ///
    const Model::SuperRowVectorType& SigmaInfty() const noexcept;

    ///
    /// @brief Returns -i K, where K denotes the current kernel.
    ///
    const SciCore::ChebAdaptive<Model::SuperRowVectorType>& K() const noexcept;

    ///
    /// @brief Returns -i \Sigma_{\infty} -i K(0), where \Sigma_{\infty} denotes infinite temperature current kernel and K(0) the current kernel at zero frequency.
    ///
    Model::SuperRowVectorType zeroFrequency() const;

    ///
    /// @brief Returns the stationary current.
    ///
    SciCore::Real stationaryCurrent(const Model::OperatorType& stationaryState) const;

    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(_model, _errorGoal, _minusISigmaInfty, _minusIK);
    }

  private:
    std::unique_ptr<Model> _model;
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

REALTIMETRANSPORT_EXPORT SciCore::ChebAdaptive<SciCore::Real> computeCurrent(
    const RenormalizedPT::CurrentKernel& KCurrent,
    const Propagator& propagator,
    const Model::OperatorType& rho0);

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_RENORMALIZED_PT_CURRENT_KERNEL_H
