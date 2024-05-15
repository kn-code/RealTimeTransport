//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_DOUBLE_DOT_H
#define REAL_TIME_TRANSPORT_DOUBLE_DOT_H

#include "../Model.h"

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/polymorphic.hpp>

#include <SciCore/Definitions.h>
#include <SciCore/Serialization.h>

namespace RealTimeTransport
{

// Double dot without spin
// Basis for operators: |00>, |10>=d_1^\dagger |00>, |01> = d_2^\dagger |00>, |11>=d_1^\dagger d_2^\dagger |00>
class REALTIMETRANSPORT_EXPORT DoubleDot final : public Model
{
  public:
    DoubleDot() noexcept                             = default;
    DoubleDot(DoubleDot&& other) noexcept            = default;
    DoubleDot(const DoubleDot& other)                = default;
    DoubleDot& operator=(DoubleDot&& other) noexcept = default;
    DoubleDot& operator=(const DoubleDot& other)     = default;

    DoubleDot(
        SciCore::Real epsilon1,
        SciCore::Real epsilon2,
        SciCore::Real U,
        SciCore::Complex t,
        const SciCore::RealVector& T,
        const SciCore::RealVector& mu,
        const SciCore::RealVector& Gamma1,
        const SciCore::RealVector& Gamma2);

    ///
    /// @brief The Hilbert space dimension is 4.
    ///
    int dimHilbertSpace() const noexcept override
    {
        return 4;
    }

    ///
    /// @brief Two single particle states.
    ///
    int numStates() const noexcept override
    {
        return 2;
    }

    ///
    /// @brief Reservoir don't have spin, thus returns 1.
    ///
    int numChannels() const noexcept override
    {
        return 1;
    }

    ///
    /// @brief Returns the number of reservoirs the system is connected to.
    ///
    int numReservoirs() const override
    {
        return _temperatures.size();
    }

    ///
    /// @brief Returns the Hamiltonian in the basis |0>, |↑>, |↓>, |↑↓>.
    ///
    OperatorType H() const override
    {
        OperatorType n1 = d(0).adjoint() * d(0);
        OperatorType n2 = d(1).adjoint() * d(1);

        return _epsilon1 * n1 + _epsilon2 * n2 + _u * n1 * n2 + _t * d(0).adjoint() * d(1) +
               std::conj(_t) * d(1).adjoint() * d(0);
    }

    ///
    /// @brief Returns the annihilation operator of the single particle state indexed by l=0, 1.
    ///
    OperatorType d(int l) const override;

    ///
    /// @brief Returns the coupling coefficient in the tunneling Hamiltonian. l represents the dot index. nu is not used in this model.
    ///
    SciCore::Complex coupling(int r, int nu, int l) const override;

    ///
    /// @brief Returns the temperatures of the reservoirs the system is connected to.
    ///
    const SciCore::RealVector& temperatures() const noexcept override
    {
        return _temperatures;
    }

    ///
    /// @brief Returns the chemical potentials of the reservoirs the system is connected to.
    ///
    const SciCore::RealVector& chemicalPotentials() const noexcept override
    {
        return _chemicalPotentials;
    }

    SupervectorType vectorize(const OperatorType& op) const override;
    OperatorType operatorize(const SupervectorType& supervector) const override;
    const std::vector<int>& blockDimensions() const noexcept override;

    std::unique_ptr<Model> copy() const override;

    SciCore::Real epsilon1() const noexcept
    {
        return _epsilon2;
    }

    SciCore::Real epsilon2() const noexcept
    {
        return _epsilon2;
    }

    SciCore::Real U() const noexcept
    {
        return _u;
    }

    const SciCore::RealVector& Gamma1() const noexcept
    {
        return _gamma1;
    }

    const SciCore::RealVector& Gamma2() const noexcept
    {
        return _gamma2;
    }

    bool isEqual(const Model& other) const override
    {
        const DoubleDot& rhs = dynamic_cast<const DoubleDot&>(other);
        return (_epsilon1 == rhs._epsilon1) && (_epsilon2 == rhs._epsilon2) && (_u == rhs._u) && (_t == rhs._t) &&
               (_temperatures == rhs._temperatures) && (_chemicalPotentials == rhs._chemicalPotentials) &&
               (_gamma1 == rhs._gamma1) && (_gamma2 == rhs._gamma2);
    }

    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(_epsilon1, _epsilon2, _u, _t, _temperatures, _chemicalPotentials, _gamma1, _gamma2);
    }

  private:
    SciCore::Real _epsilon1 = 0.0;
    SciCore::Real _epsilon2 = 0.0;
    SciCore::Real _u        = 0.0;
    SciCore::Complex _t     = 0.0;
    SciCore::RealVector _temperatures;
    SciCore::RealVector _chemicalPotentials;
    SciCore::RealVector _gamma1;
    SciCore::RealVector _gamma2;

    static const std::vector<int> _blockDimensions;
};

} // namespace RealTimeTransport

// FIXME register type

#endif // REAL_TIME_TRANSPORT_DOUBLE_DOT_H
