//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

///
/// \file   Model.h
///
/// \brief  Generic model interface.
///

#ifndef REAL_TIME_TRANSPORT_MODEL_H
#define REAL_TIME_TRANSPORT_MODEL_H

#include "RealTimeTransport_export.h"

#include <memory>

#include <SciCore/Definitions.h>

#include "BlockMatrices/BlockDiagonalMatrix.h"
#include "BlockMatrices/BlockMatrix.h"

namespace RealTimeTransport
{

///
///  \defgroup Models Models
///
///  \brief Contains classes and methods related to the implementation of different models.
///
///  This page describes classes and methods related to the implementation of different models. A model is defined by
///  a class that inherits from the abstract \ref Model class. This requires one to implement several functions, which
///  provide, for example, the Hamiltonian, the field operators and information about the block structure
///  of the memory kernel.
///
///  Models should be instantiated using the \ref createModel function, which automatically takes care of the involved
///  memory management.
///
///  \{
///

///
/// @brief Abstract class representing a model.
///
/// This class defines the abstract interface from which all models need to inherit.
///
class REALTIMETRANSPORT_EXPORT Model
{
  public:
    /// @brief Type of ordinary operators.
    using OperatorType = SciCore::Matrix;

    /// @brief Type of vectorized operators.
    using SupervectorType = SciCore::Vector;

    /// @brief Type of adjoint of vectorized operators.
    using SuperRowVectorType = SciCore::RowVector;

    /// @brief Type of superoperators.
    using SuperoperatorType = SciCore::Matrix;

    /// @brief Type representing superfermions.
    using SuperfermionType = BlockMatrix;

    Model() noexcept;
    virtual ~Model() noexcept;

    ///
    /// @brief  Equality comparison operator
    ///
    bool operator==(const Model& other) const;

    ///
    /// @brief  Inequality comparison operator
    ///
    bool operator!=(const Model& other) const;

    ///
    /// @brief Returns the dimension of the Hilbert space.
    ///
    virtual int dimHilbertSpace() const noexcept = 0;

    ///
    /// @brief Returns the number of single particle states (including spin).
    ///
    virtual int numStates() const noexcept = 0;

    ///
    /// @brief Returns the number of channels in each reservoir (e.g. spin).
    ///
    virtual int numChannels() const noexcept = 0;

    ///
    /// @brief Returns the number of reservoirs the system is connected to.
    ///
    virtual int numReservoirs() const = 0;

    ///
    /// @brief Returns the Hamiltonian.
    ///
    virtual OperatorType H() const = 0;

    ///
    /// @brief Returns the parity operator.
    ///
    virtual OperatorType P() const;

    ///
    /// @brief Returns the annihilation operator of the single particle state indexed by l.
    ///
    virtual OperatorType d(int l) const = 0;

    ///
    /// @brief Returns the coupling coefficient in the tunneling Hamiltonian, given by \f$\sqrt{\lambda}_{r \nu} t_{r \nu l}\f$.
    ///
    virtual SciCore::Complex coupling(int r, int nu, int l) const = 0;

    ///
    /// @brief Returns the temperatures of the reservoirs the system is connected to.
    ///
    virtual const SciCore::RealVector& temperatures() const noexcept = 0;

    ///
    /// @brief Returns the chemical potentials of the reservoirs the system is connected to.
    ///
    virtual const SciCore::RealVector& chemicalPotentials() const noexcept = 0;

    ///
    /// @brief Vectorizes an operator into its supervector form.
    ///
    virtual SupervectorType vectorize(const OperatorType& op) const;

    ///
    /// @brief Transforms a supervector back into operator form.
    ///
    virtual OperatorType operatorize(const SupervectorType& supervector) const;

    ///
    /// @brief The memory kernel and propagator are block diagonal.
    ///        This function returns a reference to a vector containing the dimensions of these blocks.
    ///
    virtual const std::vector<int>& blockDimensions() const noexcept = 0;

    ///
    /// @brief Returns a deep copy of the model.
    ///
    virtual std::unique_ptr<Model> copy() const = 0;

    ///
    /// @brief Returns true if other is the same as *this, otherwise false.
    ///
    virtual bool isEqual(const Model& other) const = 0;
};

///
/// @brief Creates and returns a memory managed Model object.
///
/// @param params   List of parameters to pass to the constructor.
///
template <typename ConcreteModelType, typename... Params>
    requires std::is_base_of_v<Model, ConcreteModelType>
std::unique_ptr<Model> createModel(Params&&... params)
{
    return std::make_unique<ConcreteModelType>(std::forward<Params>(params)...);
}

/// \} // end of Models

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_MODEL_H
