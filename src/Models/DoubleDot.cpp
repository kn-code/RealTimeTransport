//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/Models/DoubleDot.h"

#include <cereal/types/complex.hpp>

namespace RealTimeTransport
{

const std::vector<int> DoubleDot::_blockDimensions{6, 1, 1, 4, 4};

DoubleDot::DoubleDot(
    SciCore::Real epsilon1,
    SciCore::Real epsilon2,
    SciCore::Real U,
    SciCore::Complex Omega,
    const SciCore::RealVector& T,
    const SciCore::RealVector& mu,
    const SciCore::RealVector& Gamma1,
    const SciCore::RealVector& Gamma2)
    : _epsilon1(epsilon1), _epsilon2(epsilon2), _u(U), _omega(Omega), _temperatures(T), _chemicalPotentials(mu),
      _gamma1(Gamma1), _gamma2(Gamma2)
{
}

SciCore::Real DoubleDot::epsilon1() const noexcept
{
    return _epsilon2;
}

SciCore::Real DoubleDot::epsilon2() const noexcept
{
    return _epsilon2;
}

SciCore::Real DoubleDot::U() const noexcept
{
    return _u;
}

const SciCore::RealVector& DoubleDot::Gamma1() const noexcept
{
    return _gamma1;
}

const SciCore::RealVector& DoubleDot::Gamma2() const noexcept
{
    return _gamma2;
}

int DoubleDot::dimHilbertSpace() const noexcept
{
    return 4;
}

int DoubleDot::numStates() const noexcept
{
    return 2;
}

int DoubleDot::numChannels() const noexcept
{
    return 1;
}

int DoubleDot::numReservoirs() const
{
    return _temperatures.size();
}

DoubleDot::OperatorType DoubleDot::H() const
{
    OperatorType n1 = d(0).adjoint() * d(0);
    OperatorType n2 = d(1).adjoint() * d(1);

    return _epsilon1 * n1 + _epsilon2 * n2 + _u * n1 * n2 + _omega * d(0).adjoint() * d(1) +
           std::conj(_omega) * d(1).adjoint() * d(0);
}

DoubleDot::OperatorType DoubleDot::d(int l) const
{
    if (l == 0)
    {
        return OperatorType{
            {0, 1, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 1},
            {0, 0, 0, 0}
        };
    }

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

SciCore::Complex DoubleDot::coupling(int r, int, int l) const
{
    using namespace SciCore;

    const Real pi = std::numbers::pi_v<Real>;

    if (l == 0)
    {
        return std::sqrt(_gamma1[r] / (2 * pi));
    }
    else
    {
        return std::sqrt(_gamma2[r] / (2 * pi));
    }
}

Model::SupervectorType DoubleDot::vectorize(const Model::OperatorType& op) const
{
    // clang-format off
    // Reordering the basis vectors in this way makes the memory kernel block diagonal.
    return SupervectorType{{
        // ΔN = 0
        op(0, 0), // |00><00|
        op(1, 1), // |10><10|
        op(2, 2), // |01><01|
        op(3, 3), // |11><11|
        op(1, 2), // |10><01|
        op(2, 1), // |01><10|
        // ΔN = 2
        op(3, 0), // |11><00|
        // ΔN = -2
        op(0, 3), // |00><11|
        // ΔN = 1
        op(1, 0), // |10><00|
        op(3, 2), // |11><01|
        op(2, 0), // |01><00|
        op(3, 1), // |11><10|
        // ΔN = -1
        op(0, 2), // |00><01|
        op(1, 3), // |10><11|
        op(0, 1), // |00><10|
        op(2, 3), // |01><11|
    }};
    // clang-format on
}

Model::OperatorType DoubleDot::operatorize(const Model::SupervectorType& v) const
{
    return OperatorType{
        { v[0], v[14], v[12],  v[7]},
        { v[8],  v[1],  v[4], v[13]},
        {v[10],  v[5],  v[2], v[15]},
        { v[6], v[11],  v[9],  v[3]}
    };
}

const std::vector<int>& DoubleDot::blockDimensions() const noexcept
{
    return _blockDimensions;
}

std::unique_ptr<Model> DoubleDot::copy() const
{
    return std::make_unique<DoubleDot>(*this);
}

void DoubleDot::serialize(cereal::BinaryInputArchive& archive)
{
    archive(_epsilon1, _epsilon2, _u, _omega, _temperatures, _chemicalPotentials, _gamma1, _gamma2);
}

void DoubleDot::serialize(cereal::BinaryOutputArchive& archive)
{
    archive(_epsilon1, _epsilon2, _u, _omega, _temperatures, _chemicalPotentials, _gamma1, _gamma2);
}

void DoubleDot::serialize(cereal::PortableBinaryInputArchive& archive)
{
    archive(_epsilon1, _epsilon2, _u, _omega, _temperatures, _chemicalPotentials, _gamma1, _gamma2);
}

void DoubleDot::serialize(cereal::PortableBinaryOutputArchive& archive)
{
    archive(_epsilon1, _epsilon2, _u, _omega, _temperatures, _chemicalPotentials, _gamma1, _gamma2);
}

void DoubleDot::serialize(cereal::JSONInputArchive& archive)
{
    archive(_epsilon1, _epsilon2, _u, _omega, _temperatures, _chemicalPotentials, _gamma1, _gamma2);
}

void DoubleDot::serialize(cereal::JSONOutputArchive& archive)
{
    archive(_epsilon1, _epsilon2, _u, _omega, _temperatures, _chemicalPotentials, _gamma1, _gamma2);
}

} // namespace RealTimeTransport
