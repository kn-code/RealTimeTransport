//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_ANDERSON_DOT_H
#define REAL_TIME_TRANSPORT_ANDERSON_DOT_H

#include "../Model.h"

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>

#include <SciCore/Definitions.h>
#include <SciCore/Serialization.h>

namespace RealTimeTransport
{

// Basis for operators: |0>, |↑>, |↓>, |↑↓> = d^\dagger_\uparrow d^\dagger_\downarrow |0>
class REALTIMETRANSPORT_EXPORT AndersonDot final : public Model
{
  public:
    AndersonDot() noexcept                               = default;
    AndersonDot(AndersonDot&& other) noexcept            = default;
    AndersonDot(const AndersonDot& other)                = default;
    AndersonDot& operator=(AndersonDot&& other) noexcept = default;
    AndersonDot& operator=(const AndersonDot& other)     = default;

    AndersonDot(
        SciCore::Real epsilon,
        SciCore::Real B,
        SciCore::Real U,
        const SciCore::RealVector& T,
        const SciCore::RealVector& mu,
        const SciCore::RealVector& Gamma);

    AndersonDot(
        SciCore::Real epsilon,
        SciCore::Real B,
        SciCore::Real U,
        const SciCore::RealVector& T,
        const SciCore::RealVector& mu,
        const SciCore::RealVector& GammaUp,
        const SciCore::RealVector& GammaDown);

    ///
    /// @brief The Hilbert space dimension is 4.
    ///
    int dimHilbertSpace() const noexcept override;

    ///
    /// @brief The single particle states are ↑ and ↓, thus returns 2.
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
    /// @brief Returns the Hamiltonian in the basis |0>, |↑>, |↓>, |↑↓>.
    ///
    OperatorType H() const override;

    ///
    /// @brief Returns the annihilation operator of the single particle state indexed by l=up (0), down(1).
    ///
    OperatorType d(int l) const override;

    ///
    /// @brief Returns the coupling coefficient in the tunneling Hamiltonian. Both nu and l represent spin degrees of freedom.
    ///
    SciCore::Complex coupling(int r, int nu, int l) const override;

    ///
    /// @brief Returns the temperatures of the reservoirs the system is connected to.
    ///
    const SciCore::RealVector& temperatures() const noexcept override;

    ///
    /// @brief Returns the chemical potentials of the reservoirs the system is connected to.
    ///
    const SciCore::RealVector& chemicalPotentials() const noexcept override;

    SupervectorType vectorize(const OperatorType& op) const override;
    OperatorType operatorize(const SupervectorType& supervector) const override;
    const std::vector<int>& blockDimensions() const noexcept override;

    std::unique_ptr<Model> copy() const override;

    SciCore::Real epsilon() const noexcept;
    SciCore::Real B() const noexcept;
    SciCore::Real U() const noexcept;

    const SciCore::RealVector& GammaUp() const noexcept;
    const SciCore::RealVector& GammaDown() const noexcept;

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
