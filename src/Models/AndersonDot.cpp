//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/Models/AndersonDot.h"

#include <fstream>

namespace RealTimeTransport
{

const std::vector<int> AndersonDot::_blockDimensions{4, 1, 1, 1, 1, 2, 2, 2, 2};

AndersonDot::AndersonDot(
    SciCore::Real epsilon,
    SciCore::Real B,
    SciCore::Real U,
    const SciCore::RealVector& T,
    const SciCore::RealVector& mu,
    const SciCore::RealVector& Gamma)
    : AndersonDot(epsilon, B, U, T, mu, Gamma, Gamma)
{
}

AndersonDot::AndersonDot(
    SciCore::Real epsilon,
    SciCore::Real B,
    SciCore::Real U,
    const SciCore::RealVector& T,
    const SciCore::RealVector& mu,
    const SciCore::RealVector& GammaUp,
    const SciCore::RealVector& GammaDown)
    : _epsilon(epsilon), _b(B), _u(U), _temperatures(T), _chemicalPotentials(mu), _gammaUp(GammaUp),
      _gammaDown(GammaDown)
{
}

SciCore::Real AndersonDot::epsilon() const noexcept
{
    return _epsilon;
}

SciCore::Real AndersonDot::B() const noexcept
{
    return _b;
}

SciCore::Real AndersonDot::U() const noexcept
{
    return _u;
}

const SciCore::RealVector& AndersonDot::GammaUp() const noexcept
{
    return _gammaUp;
}

const SciCore::RealVector& AndersonDot::GammaDown() const noexcept
{
    return _gammaDown;
}

int AndersonDot::dimHilbertSpace() const noexcept
{
    return 4;
}

int AndersonDot::numStates() const noexcept
{
    return 2;
}

int AndersonDot::numChannels() const noexcept
{
    return 2;
}

int AndersonDot::numReservoirs() const
{
    return _temperatures.size();
}

AndersonDot::OperatorType AndersonDot::H() const
{
    OperatorType nUp   = d(0).adjoint() * d(0);
    OperatorType nDown = d(1).adjoint() * d(1);

    return _epsilon * (nUp + nDown) + _b / 2 * (nUp - nDown) + _u * nUp * nDown;
}

// Basis |0>, |↑>, |↓>, |↑↓> = d^\dagger_\uparrow d^\dagger_\downarrow |0>
AndersonDot::OperatorType AndersonDot::d(int l) const
{
    // Spin up
    if (l == 0)
    {
        return OperatorType{
            {0, 1, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 1},
            {0, 0, 0, 0}
        };
    }

    // Spin down
    else
    {
        return OperatorType{
            {0, 0, 1,  0},
            {0, 0, 0, -1},
            {0, 0, 0,  0},
            {0, 0, 0,  0}
        };
    }
}

SciCore::Complex AndersonDot::coupling(int r, int nu, int l) const
{
    using namespace SciCore;

    // Spin conservation
    if (nu != l)
    {
        return 0.0;
    }

    const Real pi = std::numbers::pi_v<Real>;

    if (l == 0) // Spin up
    {
        return std::sqrt(_gammaUp[r] / (2 * pi));
    }
    else // Spin down
    {
        return std::sqrt(_gammaDown[r] / (2 * pi));
    }
}

const SciCore::RealVector& AndersonDot::temperatures() const noexcept
{
    return _temperatures;
}

const SciCore::RealVector& AndersonDot::chemicalPotentials() const noexcept
{
    return _chemicalPotentials;
}

Model::SupervectorType AndersonDot::vectorize(const Model::OperatorType& op) const
{
    // clang-format off
    // Reordering the basis vectors in this way makes the memory kernel block diagonal.
    return SupervectorType{{
        // ΔN = 0, ΔS = 0
        op(0, 0), // |0><0|
        op(1, 1), // |↑><↑|
        op(2, 2), // |↓><↓|
        op(3, 3), // |↑↓><↑↓|
        // ΔN = 0, ΔS = 1
        op(1, 2), // |↑><↓|
        // ΔN = 0, ΔS = -1
        op(2, 1), // |↓><↑|
        // ΔN = 2, ΔS = 0
        op(3, 0), // |↑↓><0|
        // ΔN = -2, ΔS = 0
        op(0, 3), // |0><↑↓|
        // ΔN = 1, ΔS = 0.5
        op(1, 0), // |↑><0|
        op(3, 2), // |↑↓><↓|
        // ΔN = 1, ΔS = -0.5
        op(2, 0), // |↓><0|
        op(3, 1), // |↑↓><↑|
        // ΔN = -1, ΔS = 0.5
        op(0, 2), // |0><↓|
        op(1, 3), // |↑><↑↓|
        // ΔN = -1, ΔS = -0.5
        op(0, 1), // |0><↑|
        op(2, 3), // |↓><↑↓|
    }};
    // clang-format on
}

Model::OperatorType AndersonDot::operatorize(const Model::SupervectorType& v) const
{
    return OperatorType{
        { v[0], v[14], v[12],  v[7]},
        { v[8],  v[1],  v[4], v[13]},
        {v[10],  v[5],  v[2], v[15]},
        { v[6], v[11],  v[9],  v[3]}
    };
}

const std::vector<int>& AndersonDot::blockDimensions() const noexcept
{
    return _blockDimensions;
}

std::unique_ptr<Model> AndersonDot::copy() const
{
    return std::make_unique<AndersonDot>(*this);
}

bool AndersonDot::isEqual(const Model& other) const
{
    const AndersonDot& rhs = dynamic_cast<const AndersonDot&>(other);
    return (_epsilon == rhs._epsilon) && (_b == rhs._b) && (_u == rhs._u) && (_temperatures == rhs._temperatures) &&
           (_chemicalPotentials == rhs._chemicalPotentials) && (_gammaUp == rhs._gammaUp) &&
           (_gammaDown == rhs._gammaDown);
}

} // namespace RealTimeTransport
