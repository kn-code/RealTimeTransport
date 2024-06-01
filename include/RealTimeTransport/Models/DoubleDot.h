//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

///
/// \file   DoubleDot.h
///
/// \brief  Implements a spinless double quantum dot.
///

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

///
/// @ingroup Models
///
/// @brief Implements a spinless double quantum dot.
///
/// This class implements a spinless double quantum dot.
/// The Hamiltonian is given by
/// \f[
///     H = \epsilon_1 n_1 + \epsilon_2 n_2 + \Omega d_1^\dagger d_2 +  \Omega^* d_2^\dagger d_1 + U n_1 n_2.
/// \f]
/// The dots are coupled to the reservoirs via the tunneling Hamiltonian
/// \f[
///     H_T = \sum_{rl} \int d\omega \sqrt{\frac{\Gamma_{rl}}{2\pi}} (d^\dagger_l a_{rl}(\omega) + a^\dagger_{rl}(\omega) d_l).
/// \f]
/// All operators are represented in the basis \f$ \ket{00}, \ket{10}=d^\dagger_1 \ket{00}, \ket{01}=d^\dagger_2 \ket{00}, \ket{11} = d^\dagger_1 d^\dagger_2 \ket{00} \f$.
///
class REALTIMETRANSPORT_EXPORT DoubleDot final : public Model
{
  public:
    /// @brief Default constructor.
    DoubleDot() noexcept = default;

    /// @brief Default move constructor.
    DoubleDot(DoubleDot&& other) noexcept = default;

    /// @brief Default copy constructor.
    DoubleDot(const DoubleDot& other) = default;

    /// @brief Default move assignment operator.
    DoubleDot& operator=(DoubleDot&& other) noexcept = default;

    /// @brief Default copy assignment operator.
    DoubleDot& operator=(const DoubleDot& other) = default;

    ///
    /// @brief Constructs a new double dot object.
    ///
    /// @param epsilon1     Energy of dot 1
    /// @param epsilon2     Energy of dot 2
    /// @param U            Coulomb interaction between the dots
    /// @param Omega        Hybridization between the dots
    /// @param T            Temperatures of the reservoirs
    /// @param mu           Chemical potentials of the reservoirs
    /// @param Gamma1       Coupling \f$ \Gamma_{r1} \f$ between dot 1 and the reservoirs
    /// @param Gamma2       Coupling \f$ \Gamma_{r2} \f$ between dot 2 and the reservoirs
    ///
    DoubleDot(
        SciCore::Real epsilon1,
        SciCore::Real epsilon2,
        SciCore::Real U,
        SciCore::Complex Omega,
        const SciCore::RealVector& T,
        const SciCore::RealVector& mu,
        const SciCore::RealVector& Gamma1,
        const SciCore::RealVector& Gamma2);

    ///
    /// @brief Returns the energy of dot 1.
    ///
    SciCore::Real epsilon1() const noexcept;

    ///
    /// @brief Returns the energy of dot 2.
    ///
    SciCore::Real epsilon2() const noexcept;

    ///
    /// @brief Returns the Coulomb interaction.
    ///
    SciCore::Real U() const noexcept;

    ///
    /// @brief Returns the coupling \f$ \Gamma_{r1} \f$ between dot 1 and the reservoirs.
    ///
    const SciCore::RealVector& Gamma1() const noexcept;

    ///
    /// @brief Returns the coupling \f$ \Gamma_{r2} \f$ between dot 2 and the reservoirs.
    ///
    const SciCore::RealVector& Gamma2() const noexcept;

    ///
    /// @brief The Hilbert space dimension is 4.
    ///
    int dimHilbertSpace() const noexcept override;

    ///
    /// @brief There are two single particle states.
    ///
    int numStates() const noexcept override;

    ///
    /// @brief Returns 1 (reservoirs are spinless).
    ///
    int numChannels() const noexcept override;

    ///
    /// @brief Returns the number of reservoirs the system is connected to.
    ///
    int numReservoirs() const override;

    ///
    /// @brief Returns the Hamiltonian.
    ///
    OperatorType H() const override;

    ///
    /// @brief Returns the annihilation operator of the dot \f$l=0, 1\f$.
    ///
    OperatorType d(int l) const override;

    ///
    /// @brief Returns the coupling coefficient in the tunneling Hamiltonian. l represents the dot index. nu is not used in this model.
    ///
    SciCore::Complex coupling(int r, int nu, int l) const override;

    const SciCore::RealVector& temperatures() const noexcept override
    {
        return _temperatures;
    }

    const SciCore::RealVector& chemicalPotentials() const noexcept override
    {
        return _chemicalPotentials;
    }

    SupervectorType vectorize(const OperatorType& op) const override;
    OperatorType operatorize(const SupervectorType& supervector) const override;
    const std::vector<int>& blockDimensions() const noexcept override;

    std::unique_ptr<Model> copy() const override;

    bool isEqual(const Model& other) const override
    {
        const DoubleDot& rhs = dynamic_cast<const DoubleDot&>(other);
        return (_epsilon1 == rhs._epsilon1) && (_epsilon2 == rhs._epsilon2) && (_u == rhs._u) &&
               (_omega == rhs._omega) && (_temperatures == rhs._temperatures) &&
               (_chemicalPotentials == rhs._chemicalPotentials) && (_gamma1 == rhs._gamma1) && (_gamma2 == rhs._gamma2);
    }

    void serialize(cereal::BinaryInputArchive& archive);
    void serialize(cereal::BinaryOutputArchive& archive);
    void serialize(cereal::PortableBinaryInputArchive& archive);
    void serialize(cereal::PortableBinaryOutputArchive& archive);
    void serialize(cereal::JSONInputArchive& archive);
    void serialize(cereal::JSONOutputArchive& archive);

  private:
    SciCore::Real _epsilon1 = 0.0;
    SciCore::Real _epsilon2 = 0.0;
    SciCore::Real _u        = 0.0;
    SciCore::Complex _omega = 0.0;
    SciCore::RealVector _temperatures;
    SciCore::RealVector _chemicalPotentials;
    SciCore::RealVector _gamma1;
    SciCore::RealVector _gamma2;

    static const std::vector<int> _blockDimensions;
};

} // namespace RealTimeTransport

CEREAL_REGISTER_TYPE(RealTimeTransport::DoubleDot)
CEREAL_REGISTER_POLYMORPHIC_RELATION(RealTimeTransport::Model, RealTimeTransport::DoubleDot)

#endif // REAL_TIME_TRANSPORT_DOUBLE_DOT_H
