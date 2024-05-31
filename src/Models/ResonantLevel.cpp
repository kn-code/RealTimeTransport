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

SciCore::Real ResonantLevel::epsilon() const noexcept
{
    return _epsilon;
}

int ResonantLevel::dimHilbertSpace() const noexcept
{
    return 2;
}

int ResonantLevel::numStates() const noexcept
{
    return 1;
}

int ResonantLevel::numChannels() const noexcept
{
    return 1;
}

int ResonantLevel::numReservoirs() const
{
    return _temperatures.size();
}

ResonantLevel::OperatorType ResonantLevel::H() const
{
    OperatorType n = d(0).adjoint() * d(0);

    return _epsilon * n;
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

const SciCore::RealVector& ResonantLevel::temperatures() const noexcept
{
    return _temperatures;
}

const SciCore::RealVector& ResonantLevel::chemicalPotentials() const noexcept
{
    return _chemicalPotentials;
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

bool ResonantLevel::isEqual(const Model& other) const
{
    const ResonantLevel& rhs = dynamic_cast<const ResonantLevel&>(other);
    return (_epsilon == rhs._epsilon) && (_temperatures == rhs._temperatures) &&
           (_chemicalPotentials == rhs._chemicalPotentials) && (_gamma == rhs._gamma);
}

} // namespace RealTimeTransport
