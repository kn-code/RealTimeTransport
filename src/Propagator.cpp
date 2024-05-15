//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include <utility>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/polymorphic.hpp>

#include <SciCore/Serialization.h>

#include "RealTimeTransport/Propagator.h"

namespace RealTimeTransport
{

Propagator::Propagator() noexcept
{
}

Propagator::Propagator(Propagator&& other) noexcept
    : _model(std::move(other._model)), _propagator(std::move(other._propagator))
{
}

Propagator::Propagator(const Propagator& other) : _model(other._model->copy()), _propagator(other._propagator)
{
}

Propagator& Propagator::operator=(Propagator&& other) noexcept
{
    _model      = std::move(other._model);
    _propagator = std::move(other._propagator);
    return *this;
}

Propagator& Propagator::operator=(const Propagator& other)
{
    _model      = other._model->copy();
    _propagator = other._propagator;
    return *this;
}

Propagator::Propagator(const std::unique_ptr<Model>& model, BlockDiagonalCheb&& Pi)
    : _model(model->copy()), _propagator(std::move(Pi))
{
}

Propagator::Propagator(const Model* model, BlockDiagonalCheb&& Pi) : _model(model->copy()), _propagator(std::move(Pi))
{
}

Model::OperatorType Propagator::operator()(SciCore::Real t, const Model::OperatorType& rho0) const
{
    return _model->operatorize(_propagator(t) * _model->vectorize(rho0));
}

std::vector<Model::OperatorType> Propagator::operator()(const SciCore::RealVector& t, const Model::OperatorType& rho0)
    const
{
    std::vector<Model::OperatorType> returnValue(t.size());
    for (int i = 0; i < t.size(); ++i)
    {
        returnValue[i] = this->operator()(t[i], rho0);
    }
    return returnValue;
}

const Model* Propagator::model() const noexcept
{
    return _model.get();
}

Propagator Propagator::diff() const
{
    return Propagator(_model, _propagator.diff());
}

BlockDiagonalCheb& Propagator::Pi() noexcept
{
    return _propagator;
}

const BlockDiagonalCheb& Propagator::Pi() const noexcept
{
    return _propagator;
}

void Propagator::serialize(cereal::BinaryInputArchive& archive)
{
    archive(_model, _propagator);
}

void Propagator::serialize(cereal::BinaryOutputArchive& archive)
{
    archive(_model, _propagator);
}

void Propagator::serialize(cereal::PortableBinaryInputArchive& archive)
{
    archive(_model, _propagator);
}

void Propagator::serialize(cereal::PortableBinaryOutputArchive& archive)
{
    archive(_model, _propagator);
}

} // namespace RealTimeTransport
