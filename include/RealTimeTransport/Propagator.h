//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_PROPAGATOR_H
#define REAL_TIME_TRANSPORT_PROPAGATOR_H

#include <vector>

#include "BlockMatrices/BlockDiagonalCheb.h"
#include "Model.h"
#include "RealTimeTransport_export.h"

namespace RealTimeTransport
{

///
/// @brief Type representing the propagator (dynamical map).
///
class REALTIMETRANSPORT_EXPORT Propagator
{
  public:
    Propagator() noexcept;
    Propagator(Propagator&& other) noexcept;
    Propagator(const Propagator& other);
    Propagator& operator=(Propagator&& other) noexcept;
    Propagator& operator=(const Propagator& other);

    Propagator(const std::unique_ptr<Model>& model, BlockDiagonalCheb&& Pi);
    Propagator(const Model* model, BlockDiagonalCheb&& Pi);

    Model::OperatorType operator()(SciCore::Real t, const Model::OperatorType& rho0) const;
    std::vector<Model::OperatorType> operator()(const SciCore::RealVector& t, const Model::OperatorType& rho0) const;

    const Model* model() const noexcept;

    Propagator diff() const;

    BlockDiagonalCheb& Pi() noexcept;
    const BlockDiagonalCheb& Pi() const noexcept;

    void serialize(cereal::BinaryInputArchive& archive);
    void serialize(cereal::BinaryOutputArchive& archive);
    void serialize(cereal::PortableBinaryInputArchive& archive);
    void serialize(cereal::PortableBinaryOutputArchive& archive);

  private:
    std::unique_ptr<Model> _model;
    BlockDiagonalCheb _propagator;
};

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_PROPAGATOR_H
