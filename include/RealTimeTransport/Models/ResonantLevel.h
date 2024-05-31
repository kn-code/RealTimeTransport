//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

///
/// \file   ResonantLevel.h
///
/// \brief  Implements a spinless level without spin.
///

#ifndef REAL_TIME_TRANSPORT_RESONANT_LEVEL_H
#define REAL_TIME_TRANSPORT_RESONANT_LEVEL_H

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
/// @brief Implements a single level without spin.
///
/// This class implements a single spinless level.
/// The Hamiltonian is given by
/// \f[
///     H = \epsilon n,
/// \f]
/// where \f$ n=d^\dagger d \f$. It is coupled to the reservoirs via the tunneling Hamiltonian
/// \f[
///     H_T = \sum_{r} \int d\omega \sqrt{\frac{\Gamma_{r}}{2\pi}} (d^\dagger a_{r}(\omega) + a^\dagger_{r}(\omega) d).
/// \f]
/// All operators are represented in the basis \f$ \ket{0}, \ket{1} \f$.
///
class REALTIMETRANSPORT_EXPORT ResonantLevel final : public Model
{
  public:
    /// @brief Default constructor.
    ResonantLevel() noexcept = default;

    /// @brief Default move constructor.
    ResonantLevel(ResonantLevel&& other) noexcept = default;

    /// @brief Default copy constructor.
    ResonantLevel(const ResonantLevel& other) = default;

    /// @brief Default move assignment operator.
    ResonantLevel& operator=(ResonantLevel&& other) noexcept = default;

    /// @brief Default copy assignment operator.
    ResonantLevel& operator=(const ResonantLevel& other) = default;

    ///
    /// @brief  Constructs a new ResonantLevel object.
    ///
    /// @param epsilon  Level energy
    /// @param T        Temperature reservoirs
    /// @param mu       Chemical potentials reservoirs
    /// @param Gamma    Dot-reservoir coupling
    ///
    ResonantLevel(
        SciCore::Real epsilon,
        const SciCore::RealVector& T,
        const SciCore::RealVector& mu,
        const SciCore::RealVector& Gamma);

    ///
    /// @brief Returns the energy of the level.
    ///
    SciCore::Real epsilon() const noexcept;

    ///
    /// @brief The Hilbert space dimension is 2 (empty and full state).
    ///
    int dimHilbertSpace() const noexcept override;

    ///
    /// @brief Only one single particle state.
    ///
    int numStates() const noexcept override;

    ///
    /// @brief Reservoirs are not spinful, thus returns 1.
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
    /// @brief Returns the annihilation operator. \f$l\f$ is always 0.
    ///
    OperatorType d(int l) const override;

    ///
    /// @brief Returns the coupling coefficient in the tunneling Hamiltonian. Both \f$\nu\f$ and \f$l\f$ are always 0.
    ///
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
        archive(_epsilon, _temperatures, _chemicalPotentials, _gamma);
    }

  private:
    SciCore::Real _epsilon = 0.0;
    SciCore::RealVector _temperatures;
    SciCore::RealVector _chemicalPotentials;
    SciCore::RealVector _gamma;

    static const std::vector<int> _blockDimensions;
};

} // namespace RealTimeTransport

CEREAL_REGISTER_TYPE(RealTimeTransport::ResonantLevel)
CEREAL_REGISTER_POLYMORPHIC_RELATION(RealTimeTransport::Model, RealTimeTransport::ResonantLevel)

#endif // REAL_TIME_TRANSPORT_RESONANT_LEVEL_H
