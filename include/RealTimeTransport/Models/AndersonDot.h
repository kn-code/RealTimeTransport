//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

///
/// \file   AndersonDot.h
///
/// \brief  Implements the Anderson quantum dot.
///

#ifndef REAL_TIME_TRANSPORT_ANDERSON_DOT_H
#define REAL_TIME_TRANSPORT_ANDERSON_DOT_H

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
/// \ingroup Models
///
/// @brief Implements an Anderson quantum dot (single level with spin).
///
/// This class implements the Anderson quantum dot model (single level with spin).
/// The Hamiltonian of the quantum dot is given by
/// \f[
///     H = \epsilon (n_\uparrow + n_\downarrow) + \frac{B}{2}  (n_\uparrow - n_\downarrow) + U n_\uparrow n_\downarrow.
/// \f]
/// It is coupled to the reservoirs via the tunneling Hamiltonian
/// \f[
///     H_T = \sum_{r\sigma} \int d\omega \sqrt{\frac{\Gamma_{r\sigma}}{2\pi}} (d^\dagger_\sigma a_{r\sigma}(\omega) + a^\dagger_{r\sigma}(\omega) d_\sigma).
/// \f]
/// All operators are represented in the basis \f$ \ket{0}, \ket{\uparrow}=d^\dagger_\uparrow \ket{0}, \ket{\downarrow}=d^\dagger_\downarrow \ket{0}, \ket{\uparrow\downarrow} = d^\dagger_\uparrow d^\dagger_\downarrow \ket{0} \f$.
///
class REALTIMETRANSPORT_EXPORT AndersonDot final : public Model
{
  public:
    /// @brief Default constructor.
    AndersonDot() noexcept = default;

    /// @brief Default move constructor.
    AndersonDot(AndersonDot&& other) noexcept = default;

    /// @brief Default copy constructor.
    AndersonDot(const AndersonDot& other) = default;

    /// @brief Default move assignment operator.
    AndersonDot& operator=(AndersonDot&& other) noexcept = default;

    /// @brief Default copy assignment operator.
    AndersonDot& operator=(const AndersonDot& other) = default;

    ///
    /// @brief Constructs a new Anderson dot object.
    ///
    /// @param epsilon  Level energy
    /// @param B        Magnetic field
    /// @param U        Coulomb interaction
    /// @param T        Temperatures of the reservoirs
    /// @param mu       Chemical potentials of the reservoirs
    /// @param Gamma    Spin-independent dot-reservoir couplings \f$\Gamma_r=\Gamma_{r\uparrow}=\Gamma_{r\downarrow}\f$
    ///
    AndersonDot(
        SciCore::Real epsilon,
        SciCore::Real B,
        SciCore::Real U,
        const SciCore::RealVector& T,
        const SciCore::RealVector& mu,
        const SciCore::RealVector& Gamma);

    ///
    /// @brief Constructs a new Anderson dot object.
    ///
    /// @param epsilon    Level energy
    /// @param B          Magnetic field
    /// @param U          Coulomb interaction
    /// @param T          Temperatures
    /// @param mu         Chemical potentials
    /// @param GammaUp    Dot-reservoir coupling \f$\Gamma_{r\uparrow}\f$
    /// @param GammaDown  Dot-reservoir coupling \f$\Gamma_{r\downarrow}\f$
    ///
    AndersonDot(
        SciCore::Real epsilon,
        SciCore::Real B,
        SciCore::Real U,
        const SciCore::RealVector& T,
        const SciCore::RealVector& mu,
        const SciCore::RealVector& GammaUp,
        const SciCore::RealVector& GammaDown);

    ///
    /// @brief Returns the energy of the level.
    ///
    SciCore::Real epsilon() const noexcept;

    ///
    /// @brief Returns the magnetic field.
    ///
    SciCore::Real B() const noexcept;

    ///
    /// @brief Returns the Coulomb interaction.
    ///
    SciCore::Real U() const noexcept;

    ///
    /// @brief Returns the couplings \f$\Gamma_{r\uparrow}\f$.
    ///
    const SciCore::RealVector& GammaUp() const noexcept;

    ///
    /// @brief Returns the couplings \f$\Gamma_{r\downarrow}\f$.
    ///
    const SciCore::RealVector& GammaDown() const noexcept;

    ///
    /// @brief The Hilbert space dimension is 4.
    ///
    int dimHilbertSpace() const noexcept override;

    ///
    /// @brief The single particle states are \f$\uparrow\f$ and \f$\downarrow\f$, thus returns 2.
    ///
    int numStates() const noexcept override;

    ///
    /// @brief Each reservoir is spinful, thus returns 2.
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
    /// @brief Returns the annihilation operator of the single particle state indexed by \f$ l=\uparrow\equiv 0 \f$, \f$ \downarrow\equiv 1 \f$.
    ///
    OperatorType d(int l) const override;

    SciCore::Complex coupling(int r, int nu, int l) const override;
    const SciCore::RealVector& temperatures() const noexcept override;
    const SciCore::RealVector& chemicalPotentials() const noexcept override;

    SupervectorType vectorize(const OperatorType& op) const override;
    OperatorType operatorize(const SupervectorType& supervector) const override;
    const std::vector<int>& blockDimensions() const noexcept override;

    std::unique_ptr<Model> copy() const override;

    bool isEqual(const Model& other) const override;

    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(_epsilon, _b, _u, _temperatures, _chemicalPotentials, _gammaUp, _gammaDown);
    }

  private:
    SciCore::Real _epsilon = 0.0;
    SciCore::Real _b       = 0.0;
    SciCore::Real _u       = 0.0;
    SciCore::RealVector _temperatures;
    SciCore::RealVector _chemicalPotentials;
    SciCore::RealVector _gammaUp;
    SciCore::RealVector _gammaDown;

    static const std::vector<int> _blockDimensions;
};

} // namespace RealTimeTransport

CEREAL_REGISTER_TYPE(RealTimeTransport::AndersonDot)
CEREAL_REGISTER_POLYMORPHIC_RELATION(RealTimeTransport::Model, RealTimeTransport::AndersonDot)

#endif // REAL_TIME_TRANSPORT_ANDERSON_DOT_H
