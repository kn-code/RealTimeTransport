//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

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
/// @brief Single level without spin.
///
class REALTIMETRANSPORT_EXPORT ResonantLevel final : public Model
{
    // Basis for operators: |0>, |1>
  public:
    ResonantLevel() noexcept                                 = default;
    ResonantLevel(ResonantLevel&& other) noexcept            = default;
    ResonantLevel(const ResonantLevel& other)                = default;
    ResonantLevel& operator=(ResonantLevel&& other) noexcept = default;
    ResonantLevel& operator=(const ResonantLevel& other)     = default;

    ResonantLevel(
        SciCore::Real epsilon,
        const SciCore::RealVector& T,
        const SciCore::RealVector& mu,
        const SciCore::RealVector& Gamma);

    ///
    /// @brief The Hilbert space dimension is 2 (empty and full state).
    ///
    int dimHilbertSpace() const noexcept override
    {
        return 2;
    }

    ///
    /// @brief Only one single particle state
    ///
    int numStates() const noexcept override
    {
        return 1;
    }

    ///
    /// @brief Reservoirs are not spinful, thus returns 1.
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
    /// @brief Returns the Hamiltonian.
    ///
    OperatorType H() const override
    {
        OperatorType n = d(0).adjoint() * d(0);

        return _epsilon * n;
    }

    SciCore::Real epsilon() const noexcept
    {
        return _epsilon;
    }

    ///
    /// @brief Returns the annihilation operator of the single particle state indexed by l=down, up.
    ///
    OperatorType d(int l) const override;

    ///
    /// @brief Returns the coupling coefficient in the tunneling Hamiltonian. Both nu and l are always 0.
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

    bool isEqual(const Model& other) const override
    {
        const ResonantLevel& rhs = dynamic_cast<const ResonantLevel&>(other);
        return (_epsilon == rhs._epsilon) && (_temperatures == rhs._temperatures) &&
               (_chemicalPotentials == rhs._chemicalPotentials) && (_gamma == rhs._gamma);
    }

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
