//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include <fstream>

#include "RealTimeTransport/Models/ResonantLevel.h"

namespace RealTimeTransport
{

const std::vector<int> ResonantLevel::_blockDimensions{2, 2};

ResonantLevel::ResonantLevel(
    SciCore::Real epsilon,
    const SciCore::RealVector& T,
    const SciCore::RealVector& mu,
    const SciCore::RealVector& Gamma)
    : _epsilon(epsilon), _temperatures(T), _chemicalPotentials(mu), _gamma(Gamma)
{
}

// Basis |0>, |1>
ResonantLevel::OperatorType ResonantLevel::d(int) const
{
    return OperatorType{
        {0, 1},
        {0, 0}
    };
}

SciCore::Complex ResonantLevel::coupling(int r, int, int) const
{
    using namespace SciCore;

    const Real pi = std::numbers::pi_v<Real>;
    return std::sqrt(_gamma[r] / (2 * pi));
}

Model::SupervectorType ResonantLevel::vectorize(const Model::OperatorType& op) const
{
    // Reordering the basis vectors in this way makes the memory kernel block diagonal.
    return SupervectorType{
        {
         op(0, 0),
         op(1, 1),
         op(1, 0),
         op(0, 1),
         }
    };
}

Model::OperatorType ResonantLevel::operatorize(const Model::SupervectorType& v) const
{
    return OperatorType{
        {v[0], v[3]},
        {v[2], v[1]}
    };
}

const std::vector<int>& ResonantLevel::blockDimensions() const noexcept
{
    return _blockDimensions;
}

std::unique_ptr<Model> ResonantLevel::copy() const
{
    return std::make_unique<ResonantLevel>(_epsilon, _temperatures, _chemicalPotentials, _gamma);
}

} // namespace RealTimeTransport
