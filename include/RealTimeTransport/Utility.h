//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_UTILITY_H
#define REAL_TIME_TRANSPORT_UTILITY_H

#include <memory>

#include <Eigen/Eigenvalues>

#include "BlockMatrices/BlockDiagonalMatrix.h"
#include "BlockMatrices/BlockMatrix.h"
#include "Model.h"
#include "RealTimeTransport_export.h"

namespace RealTimeTransport
{

enum class Eta : int
{
    Minus = 0,
    Plus  = 1,
    _count
};

enum class Keldysh : int
{
    Minus = 0,
    Plus  = 1,
    _count
};

///
/// \brief      Computes a superfermion for a given model.
///
/// This function computes superfermionic creation/annihilation operators for given models.
/// The superfermions are superoperators defined as
/// $$
/// 	D^p_{\eta l} \bullet = \frac{1}{\sqrt{2}} ( d_{\eta l} \bullet + p (-\mathbb{1})^n \bullet (-\mathbb{1})^n d_{\eta l} )
/// $$
/// where \f$ \bullet \f$ denotes some operator argument.
///
/// \param      model   The model for which the superfermion is computed.
/// \param      p       Represents creation or annihilation.
/// \param      eta     Either "-" or "+".
/// \param      l   Index for internal degrees of freedom (always \f$ \geq 0 \f$ !).
///
REALTIMETRANSPORT_EXPORT Model::SuperfermionType computeSuperfermion(Keldysh p, Eta eta, int l, const Model* model);

REALTIMETRANSPORT_EXPORT inline Model::SuperfermionType computeSuperfermion(
    Keldysh p,
    Eta eta,
    int l,
    const std::unique_ptr<Model>& model)
{
    return computeSuperfermion(p, eta, l, model.get());
}

///
/// @brief Computes all creation or annihilation superfermions (for p=+,- respectively) for a given model.
///
REALTIMETRANSPORT_EXPORT std::vector<Model::SuperfermionType> computeAllSuperfermions(Keldysh p, const Model* model);

REALTIMETRANSPORT_EXPORT inline std::vector<Model::SuperfermionType> computeAllSuperfermions(
    Keldysh p,
    const std::unique_ptr<Model>& model)
{
    return computeAllSuperfermions(p, model.get());
}

///
/// @brief Convenience function to compute \f$ \Gamma_{η r l_1 l_2}  \f$.
///
REALTIMETRANSPORT_EXPORT SciCore::Complex computeGamma(Eta eta, int r, int l1, int l2, const Model* model);

REALTIMETRANSPORT_EXPORT inline SciCore::Complex computeGamma(
    Eta eta,
    int r,
    int l1,
    int l2,
    const std::unique_ptr<Model>& model)
{
    return computeGamma(eta, r, l1, l2, model.get());
}

///
/// @brief Returns \f$ -i L \f$ for a given specific _model_, where \f$ L = [H, \bullet] \f$ denotes the bare
/// Liouvillian.
///
REALTIMETRANSPORT_EXPORT BlockDiagonalMatrix computeLiouvillian(const Model* model);

REALTIMETRANSPORT_EXPORT inline BlockDiagonalMatrix computeLiouvillian(const std::unique_ptr<Model>& model)
{
    return computeLiouvillian(model.get());
}

///
/// @brief Returns the \f$ \delta \f$-singular part \f$ -i \Sigma_\infty \f$ of the infinite temperature memory kernel
/// for a given fermionic wideband _model_.
///
REALTIMETRANSPORT_EXPORT BlockDiagonalMatrix computeSigmaInfty(
    const std::vector<Model::SuperfermionType>& superfermion,
    const std::vector<Model::SuperfermionType>& superfermionAnnihilation,
    const Model* model);

REALTIMETRANSPORT_EXPORT inline BlockDiagonalMatrix computeSigmaInfty(
    const std::vector<Model::SuperfermionType>& superfermion,
    const std::vector<Model::SuperfermionType>& superfermionAnnihilation,
    const std::unique_ptr<Model>& model)
{
    return computeSigmaInfty(superfermion, superfermionAnnihilation, model.get());
}

///
/// @brief Returns the \f$ \delta \f$-singular part of the current kernel, \f$ -i \Sigma^{(I_r)}_{\infty} \f$, for a given fermionic wideband _model_.
///
REALTIMETRANSPORT_EXPORT Model::SuperRowVectorType computeSigmaInftyCurrent(
    int r,
    const std::vector<Model::SuperfermionType>& superfermionAnnihilation,
    const Model* model);

REALTIMETRANSPORT_EXPORT inline Model::SuperRowVectorType computeSigmaInftyCurrent(
    int r,
    const std::vector<Model::SuperfermionType>& superfermionAnnihilation,
    const std::unique_ptr<Model>& model)
{
    return computeSigmaInftyCurrent(r, superfermionAnnihilation, model.get());
}

REALTIMETRANSPORT_EXPORT std::vector<SciCore::Vector> computeZeroEigenvectors(
    const SciCore::Matrix& A,
    SciCore::Real tol);

REALTIMETRANSPORT_EXPORT Model::OperatorType computeStationaryState(
    const Model::SuperoperatorType& M,
    const Model* model);

REALTIMETRANSPORT_EXPORT inline Model::OperatorType computeStationaryState(
    const Model::SuperoperatorType& M,
    const std::unique_ptr<Model>& model)
{
    return computeStationaryState(M, model.get());
}

REALTIMETRANSPORT_EXPORT Model::OperatorType computeStationaryState(
    const BlockDiagonalMatrix& M,
    const Model* model,
    SciCore::Real tol);

REALTIMETRANSPORT_EXPORT inline Model::OperatorType computeStationaryState(
    const BlockDiagonalMatrix& M,
    const std::unique_ptr<Model>& model,
    SciCore::Real tol)
{
    return computeStationaryState(M, model.get(), tol);
}

// Computes the stationary state corresponding to a specific block A with index blockIndex.
// This is useful, for example, if it is known that the zero eigenvector always lives in a specific block
// of the memory kernel.
REALTIMETRANSPORT_EXPORT Model::OperatorType computeStationaryState(
    const SciCore::Matrix& M,
    const Model* model,
    int blockIndex,
    SciCore::Real tol);

inline Model::OperatorType computeStationaryState(
    const SciCore::Matrix& M,
    const std::unique_ptr<Model>& model,
    int blockIndex,
    SciCore::Real tol)
{
    return computeStationaryState(M, model.get(), blockIndex, tol);
}

REALTIMETRANSPORT_EXPORT SciCore::Matrix exp(const SciCore::Matrix& X);
REALTIMETRANSPORT_EXPORT BlockDiagonalMatrix exp(const BlockDiagonalMatrix& X);

REALTIMETRANSPORT_EXPORT SciCore::Matrix expm1(const SciCore::Matrix& X);
REALTIMETRANSPORT_EXPORT BlockDiagonalMatrix expm1(const BlockDiagonalMatrix& X);

// Converts the multi index i to a list of in total N single indices. The single index indicesList[i]
// can have values between 0 <= indicesList[i] < dims[i].
REALTIMETRANSPORT_EXPORT void multiIndexToList(int i, const int* dims, int N, int* indicesList) noexcept;
REALTIMETRANSPORT_EXPORT int listToMultiIndex(const int* indicesList, const int* dims, int N) noexcept;

struct REALTIMETRANSPORT_EXPORT Indices
{
    Eta eta;
    int l;
};

///
/// \brief      Converts multiple single indices into a multiindex.
///
/// Transforms the two indices \f$\eta\f$ and \f$l\f$ into a single multiindex \f$i\f$ which is
/// returned.
///
REALTIMETRANSPORT_EXPORT int singleToMultiIndex(Eta eta, int l, const Model* model);

REALTIMETRANSPORT_EXPORT inline int singleToMultiIndex(Eta eta, int l, const std::unique_ptr<Model>& model)
{
    return singleToMultiIndex(eta, l, model.get());
}

///
/// \brief      Converts a multiindex into multiple single indices.
///
/// Transforms the multiindex \f$i\f$ into two single indices \f$\eta\f$ and \f$l\f$ which are
/// returned as a tuple. The function can be used as
///
/// \code{.cpp}
/// const auto [eta, l] = multiToSingleIndices(model, i);
/// \endcode
///
REALTIMETRANSPORT_EXPORT Indices multiToSingleIndices(int i, const Model* model);

REALTIMETRANSPORT_EXPORT inline Indices multiToSingleIndices(int i, const std::unique_ptr<Model>& model)
{
    return multiToSingleIndices(i, model.get());
}

REALTIMETRANSPORT_EXPORT SciCore::Complex gammaMinus(
    SciCore::Real t,
    Eta eta,
    int l1,
    int l2,
    int r,
    const Model* model);

REALTIMETRANSPORT_EXPORT inline SciCore::Complex gammaMinus(
    SciCore::Real t,
    Eta eta,
    int l1,
    int l2,
    int r,
    const std::unique_ptr<Model>& model)
{
    return gammaMinus(t, eta, l1, l2, r, model.get());
}

REALTIMETRANSPORT_EXPORT SciCore::Complex gammaMinus(SciCore::Real t, Eta eta, int l1, int l2, const Model* model);

REALTIMETRANSPORT_EXPORT inline SciCore::Complex gammaMinus(
    SciCore::Real t,
    Eta eta,
    int l1,
    int l2,
    const std::unique_ptr<Model>& model)
{
    return gammaMinus(t, eta, l1, l2, model.get());
}

// Computes \sum_1 gamma_1(t) * G^+_1 G^+_{\bar{1}}
REALTIMETRANSPORT_EXPORT SciCore::Matrix computeGammaGG(
    int blockIndex,
    SciCore::Real t,
    const std::vector<Model::SuperfermionType>& superfermion,
    const Model* model);

REALTIMETRANSPORT_EXPORT inline SciCore::Matrix computeGammaGG(
    int blockIndex,
    SciCore::Real t,
    const std::vector<Model::SuperfermionType>& superfermion,
    const std::unique_ptr<Model>& model)
{
    return computeGammaGG(blockIndex, t, superfermion, model.get());
}

// Computes \sum_1 gamma_1(t) * G^+_1 G^+_{\bar{1}}
REALTIMETRANSPORT_EXPORT BlockDiagonalMatrix
computeGammaGG(SciCore::Real t, const std::vector<Model::SuperfermionType>& superfermion, const Model* model);

REALTIMETRANSPORT_EXPORT inline BlockDiagonalMatrix computeGammaGG(
    SciCore::Real t,
    const std::vector<Model::SuperfermionType>& superfermion,
    const std::unique_ptr<Model>& model)
{
    return computeGammaGG(t, superfermion, model.get());
}

REALTIMETRANSPORT_EXPORT SciCore::RealVector defaultInitialChebSections(SciCore::Real tMax, SciCore::Real tCrit);

REALTIMETRANSPORT_EXPORT SciCore::Real defaultMinChebDistance(SciCore::Real tCrit, SciCore::Real errorGoal);

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_UTILITY_H
