//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/Utility.h"

#include <SciCore/BasicMath.h>
#include <SciCore/Utility.h>

#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <unsupported/Eigen/MatrixFunctions>

#include "RealTimeTransport/BlockMatrices/MatrixOperations.h"
#include "RealTimeTransport/Error.h"

namespace RealTimeTransport
{

BlockMatrix computeSuperfermion(Keldysh p, Eta eta, int l, const Model* model)
{
    using namespace SciCore;

    Real pEff = (p == Keldysh::Minus) ? -1 : 1;

    int dimHilbert  = model->dimHilbertSpace();
    int dimHilbert2 = dimHilbert * dimHilbert;
    Model::SuperoperatorType G(dimHilbert2, dimHilbert2);
    RealVector basisVector(dimHilbert2);

    Model::OperatorType d      = (eta == Eta::Minus) ? model->d(l) : model->d(l).adjoint();
    Model::OperatorType parity = model->P();

    for (int i = 0; i < dimHilbert2; ++i)
    {
        basisVector.setZero();
        basisVector[i]     = 1;
        auto basisOperator = model->operatorize(basisVector);

        Model::OperatorType result =
            1 / std::sqrt(Real(2)) * (d * basisOperator + pEff * parity * basisOperator * parity * d);

        G.col(i) = model->vectorize(result);
    }

    return BlockMatrix(G, model->blockDimensions());
}

std::vector<BlockMatrix> computeAllSuperfermions(Keldysh p, const Model* model)
{
    int maxIndex = 2 * model->numStates();
    std::vector<BlockMatrix> superfermions(maxIndex);
    for (int i = 0; i < maxIndex; ++i)
    {
        auto [eta, l]    = multiToSingleIndices(i, model);
        superfermions[i] = computeSuperfermion(p, eta, l, model);
    }

    return superfermions;
}

SciCore::Complex computeGamma(Eta eta, int r, int l1, int l2, const Model* model)
{
    SciCore::Real pi = std::numbers::pi_v<SciCore::Real>;

    SciCore::Complex returnValue(0, 0);
    for (int nu = 0; nu < model->numChannels(); ++nu)
    {
        returnValue += 2 * pi * model->coupling(r, nu, l1) * std::conj(model->coupling(r, nu, l2));
    }

    return (eta == Eta::Plus) ? returnValue : std::conj(returnValue);
}

BlockDiagonalMatrix computeLiouvillian(const Model* model)
{
    using namespace SciCore;

    int dim                         = model->dimHilbertSpace();
    int dim2                        = dim * dim;
    Model::OperatorType Hamiltonian = model->H();

    RealVector basisVector(dim2);
    Model::SuperoperatorType L(dim2, dim2);

    for (int i = 0; i < dim2; ++i)
    {
        basisVector.setZero();
        basisVector[i]     = 1;
        auto basisOperator = model->operatorize(basisVector);

        Model::OperatorType result = Hamiltonian * basisOperator - basisOperator * Hamiltonian;
        L.col(i)                   = model->vectorize(result);
    }

    L *= Complex(0, -1);
    BlockDiagonalMatrix returnValue(L, model->blockDimensions());
    return returnValue;
}

BlockDiagonalMatrix computeSigmaInfty(
    const std::vector<BlockMatrix>& superfermion,
    const std::vector<BlockMatrix>& superfermionAnnihilation,
    const Model* model)
{
    using namespace SciCore;

    auto minusISigmaInfty = BlockDiagonalMatrix::Zero(model->blockDimensions());

    for (int eta = 0; eta <= 1; ++eta)
    {
        for (int l1 = 0; l1 < model->numStates(); ++l1)
        {
            for (int l1Bar = 0; l1Bar < model->numStates(); ++l1Bar)
            {
                for (int r = 0; r < model->numReservoirs(); ++r)
                {
                    Complex Gamma = computeGamma(static_cast<Eta>(eta), r, l1, l1Bar, model);
                    if (Gamma != 0.0)
                    {
                        int i    = singleToMultiIndex(static_cast<Eta>(eta), l1, model);
                        int iBar = singleToMultiIndex(static_cast<Eta>(!eta), l1Bar, model);

                        addProduct(-Gamma / 2.0, superfermion[i], superfermionAnnihilation[iBar], minusISigmaInfty);
                    }
                }
            }
        }
    }

    return minusISigmaInfty;
}

Model::SuperRowVectorType computeSigmaInftyCurrent(
    int r,
    const std::vector<BlockMatrix>& superfermionAnnihilation,
    const Model* model)
{
    using namespace SciCore;
    using Operator       = Model::OperatorType;
    using Supervector    = Model::SupervectorType;
    using SuperRowVector = Model::SuperRowVectorType;

    int dim       = model->dimHilbertSpace();
    int dim2      = dim * dim;
    int numStates = model->numStates();

    Operator id          = Operator::Identity(dim, dim);
    Supervector idCol    = model->vectorize(id);
    SuperRowVector idRow = idCol.transpose();

    SuperRowVector minusISigmaInftyCurrent = SuperRowVector::Zero(dim2);
    for (int eta = 0; eta <= 1; ++eta)
    {
        for (int l1 = 0; l1 < numStates; ++l1)
        {
            for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
            {
                Complex Gamma = computeGamma(static_cast<Eta>(eta), r, l1, l1Bar, model);
                if (Gamma != 0.0)
                {
                    Real etaEff = (eta == 0) ? -1 : 1;
                    int i       = singleToMultiIndex(static_cast<Eta>(eta), l1, model);
                    int iBar    = singleToMultiIndex(static_cast<Eta>(!eta), l1Bar, model);

                    addProduct(
                        -1.0, idRow,
                        product_toDiagonal(
                            Gamma / 4.0 * etaEff, superfermionAnnihilation[i], superfermionAnnihilation[iBar]),
                        minusISigmaInftyCurrent);
                }
            }
        }
    }

    return minusISigmaInftyCurrent;
}

std::vector<SciCore::Vector> computeZeroEigenvectors(const SciCore::Matrix& A, SciCore::Real tol)
{
    using namespace SciCore;

    Eigen::ComplexEigenSolver<Matrix> es(A);
    Vector eigenvalues = es.eigenvalues();

    std::vector<SciCore::Vector> returnValue;
    for (int i = 0; i < eigenvalues.size(); ++i)
    {
        if (std::abs(eigenvalues[i]) < tol)
        {
            returnValue.emplace_back(es.eigenvectors().col(i));
        }
    }

    return returnValue;
}

Model::OperatorType computeStationaryState(const Model::SuperoperatorType& M, const Model* model)
{
    int dim2 = M.rows();
    int dim  = sqrt(dim2);

    if ((dim * dim != dim2) || (M.cols() != dim2))
    {
        throw Error("Invalid dimension");
    }

    Model::SuperoperatorType A = M;
    Model::OperatorType id     = Model::OperatorType::Identity(dim, dim);
    A.row(dim2 - 1)            = model->vectorize(id);

    Model::SupervectorType b(dim2);
    b.setZero();
    b[dim2 - 1] = 1;

    b                               = A.colPivHouseholderQr().solve(b);
    Model::OperatorType returnValue = model->operatorize(b);
    returnValue                     = (returnValue + returnValue.adjoint()) / 2.0; // Enforce strict hermicity
    return returnValue;
}

Model::OperatorType computeStationaryState(const BlockDiagonalMatrix& M, const Model* model, SciCore::Real tol)
{
    using namespace SciCore;

    Model::SupervectorType zeroEigenvector = Model::SupervectorType::Zero(M.totalRows());
    int totalZeroEigenvectors              = 0;
    int rowIndex                           = 0;
    for (int i = 0; i < M.numBlocks(); ++i)
    {
        int Mi_size                  = M(i).rows();
        std::vector<Vector> zeroVecs = computeZeroEigenvectors(M(i), tol);

        if (zeroVecs.size() > 1)
        {
            throw Error("Zero eigenvector within block is not unique");
        }
        else if (zeroVecs.size() == 1)
        {
            zeroEigenvector.segment(rowIndex, Mi_size) = zeroVecs[0];
            ++totalZeroEigenvectors;
        }

        rowIndex += Mi_size;
    }

    if (totalZeroEigenvectors == 0)
    {
        throw Error("No zero eigenvector found");
    }
    else if (totalZeroEigenvectors > 1)
    {
        throw Error("Multiple zero eigenvectors found");
    }

    Model::OperatorType returnValue  = model->operatorize(zeroEigenvector);
    returnValue                     /= returnValue.trace();

    return returnValue;
}

// Computes the stationary state. Here A is the dense matrix corresponding to the block "blockIndex".
Model::OperatorType computeStationaryState(
    const SciCore::Matrix& M,
    const Model* model,
    int blockIndex,
    SciCore::Real tol)
{
    using namespace SciCore;

    int dim2                               = std::pow(model->dimHilbertSpace(), 2);
    Model::SupervectorType zeroEigenvector = Model::SupervectorType::Zero(dim2);

    int rowIndex = 0;
    for (int i = 0; i < blockIndex; ++i)
    {
        rowIndex += model->blockDimensions()[i];
    }

    std::vector<Vector> zeroVecs = computeZeroEigenvectors(M, tol);

    if (zeroVecs.size() == 0)
    {
        throw Error("No zero eigenvector found");
    }
    else if (zeroVecs.size() > 1)
    {
        throw Error("Multiple zero eigenvectors found");
    }

    zeroEigenvector.segment(rowIndex, M.rows()) = zeroVecs[0];

    Model::OperatorType returnValue  = model->operatorize(zeroEigenvector);
    returnValue                     /= returnValue.trace();

    return returnValue;
}

SciCore::Matrix exp(const SciCore::Matrix& X)
{
    return X.exp();
}

BlockDiagonalMatrix exp(const BlockDiagonalMatrix& X)
{
    using namespace SciCore;

    int numBlocks = X.numBlocks();

    std::vector<Matrix> blocks;
    blocks.reserve(numBlocks);
    for (int i = 0; i < numBlocks; ++i)
    {
        blocks.push_back(exp(X(i)));
    }

    return BlockDiagonalMatrix(std::move(blocks));
}

SciCore::Complex expm1Helper(SciCore::Complex x, int n) noexcept
{
    using std::exp;

    if (n == 0)
    {
        return SciCore::expm1(x);
    }
    else
    {
        return std::exp(x);
    }
}

SciCore::Matrix expm1(const SciCore::Matrix& X)
{
    using namespace SciCore;

    // We could actually write std::pow(std::numeric_limits<Real>::epsilon() * 7!, 1.0 / 7.0).
    // But lets not use the tightest possible error bound for safety.
    Real bound = std::pow(std::numeric_limits<Real>::epsilon() * 720, 1.0 / 6.0);
    Real normX = SciCore::maxNorm(X);
    if (normX < bound)
    {
        auto id       = SciCore::Matrix::Identity(X.rows(), X.cols());
        Matrix taylor = X * (id + X * (id / 2. + X * (id / 6. + X * (id / 24. + X * (id / 120. + X / 720.)))));

        return taylor;
    }
    else
    {
        return X.matrixFunction(expm1Helper);
    }
}

BlockDiagonalMatrix expm1(const BlockDiagonalMatrix& X)
{
    using namespace SciCore;

    int numBlocks = X.numBlocks();

    std::vector<Matrix> blocks;
    blocks.reserve(numBlocks);
    for (int i = 0; i < numBlocks; ++i)
    {
        blocks.push_back(expm1(X(i)));
    }

    return BlockDiagonalMatrix(std::move(blocks));
}

void multiIndexToList(int i, const int* dims, int numDims, int* indicesList) noexcept
{
#ifndef NDEBUG
    int D = 1;
    for (int j = 0; j < numDims; ++j)
    {
        D *= dims[j];
    }
    assert(i < D);
#endif

    for (int j = 0; j < numDims; ++j)
    {
        indicesList[numDims - j - 1]  = i % (dims[numDims - j - 1]);
        i                            /= (dims[numDims - j - 1]);
    }
}

int listToMultiIndex(const int* indicesList, const int* dims, int numDims) noexcept
{
    constexpr int maxNumDims = 5;
    assert(numDims > 0);
    assert(numDims < maxNumDims);
#ifndef NDEBUG
    for (int j = 0; j < numDims; ++j)
    {
        assert(indicesList[j] < dims[j]);
    }
#endif

    int part_prod[maxNumDims];

    int result             = 0;
    part_prod[numDims - 1] = 1;
    for (int j = 1; j < numDims; ++j)
    {
        part_prod[numDims - j - 1]  = part_prod[numDims - j] * dims[numDims - j];
        result                     += indicesList[numDims - j - 1] * part_prod[numDims - j - 1];
    }

    return result + indicesList[numDims - 1];
}

int singleToMultiIndex(Eta eta, int l, const Model* model)
{
    int indices[2] = {(eta == Eta::Minus) ? 0 : 1, l};
    int dims[2]    = {2, model->numStates()};

    return listToMultiIndex(indices, dims, 2);
}

Indices multiToSingleIndices(int i, const Model* model)
{
    int dims[2] = {2, model->numStates()};
    int result[2];
    multiIndexToList(i, dims, 2, result);

    return Indices{(result[0] == 0) ? Eta::Minus : Eta::Plus, result[1]};
}

SciCore::Complex gammaMinus(SciCore::Real t, Eta eta, int l1, int l2, int r, const Model* model)
{
    using std::exp;
    using std::sinh;
    using namespace SciCore;

    constexpr Real pi = std::numbers::pi_v<Real>;

    Complex I(0, 1);
    Real etaEff   = (eta == Eta::Minus) ? -1 : 1;
    Real T        = model->temperatures()[r];
    Real mu       = model->chemicalPotentials()[r];
    Complex Gamma = computeGamma(eta, r, l1, l2, model);

    if (Gamma == 0.0)
    {
        return 0.0;
    }

    if (T == 0)
    {
        //return -I * (Gamma / (pi * t)) * std::exp(-I * (etaEff * mu * t));
        Gamma *= I;
        Gamma /= -pi * t;
        Gamma *= std::exp(Complex(0, -etaEff * mu * t));
        return Gamma;
    }
    else
    {
        //return -I * (T * Gamma / std::sinh(pi * t * T)) * std::exp(-I * (etaEff * mu * t));
        Gamma *= I;
        Gamma *= -T / std::sinh(pi * t * T);
        Gamma *= std::exp(Complex(0, -etaEff * mu * t));
        return Gamma;
    }
}

SciCore::Complex gammaMinus(SciCore::Real t, Eta eta, int l1, int l2, const Model* model)
{
    using namespace SciCore;
    int nRes = model->numReservoirs();

    Complex returnValue(0, 0);
    for (int r = 0; r < nRes; ++r)
    {
        returnValue += gammaMinus(t, eta, l1, l2, r, model);
    }

    return returnValue;
}

SciCore::Complex d_dmu_gammaMinus(SciCore::Real t, Eta eta, int l1, int l2, int r, const Model* model)
{
    using std::exp;
    using std::sinh;
    using namespace SciCore;

    constexpr Real pi = std::numbers::pi_v<Real>;

    Real etaEff   = (eta == Eta::Minus) ? -1 : 1;
    Real T        = model->temperatures()[r];
    Real mu       = model->chemicalPotentials()[r];
    Complex Gamma = computeGamma(eta, r, l1, l2, model);

    if (Gamma == 0.0)
    {
        return 0.0;
    }

    if (T == 0)
    {
        // return - etaEff * (Gamma / pi) * std::exp(-I * (etaEff * mu * t));
        Gamma *= -etaEff / pi;
        Gamma *= std::exp(Complex(0, -etaEff * mu * t));
        return Gamma;
    }
    else
    {
        //return - (etaEff * t * T * Gamma / std::sinh(pi * t * T)) * std::exp(-I * (etaEff * mu * t));
        Gamma *= -(etaEff * t * T) / std::sinh(pi * t * T);
        Gamma *= std::exp(Complex(0, -etaEff * mu * t));
        return Gamma;
    }
}

SciCore::Matrix computeGammaGG(
    int blockIndex,
    SciCore::Real t,
    const std::vector<BlockMatrix>& superfermion,
    const Model* model)
{
    using std::sin;
    using std::sinh;
    using namespace SciCore;

    int nRes             = model->numReservoirs();
    int numStates        = model->numStates();
    const RealVector& T  = model->temperatures();
    const RealVector& mu = model->chemicalPotentials();
    Real pi              = std::numbers::pi_v<Real>;

    int numRows        = model->blockDimensions()[blockIndex];
    Matrix returnValue = Matrix::Zero(numRows, numRows);

    if (t == 0)
    {
        for (int l1 = 0; l1 < numStates; ++l1)
        {
            for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
            {
                for (int r = 0; r < nRes; ++r)
                {
                    Complex Gamma = computeGamma(Eta::Plus, r, l1, l1Bar, model);
                    if (Gamma != 0)
                    {
                        int i    = singleToMultiIndex(Eta::Plus, l1, model);
                        int iBar = singleToMultiIndex(Eta::Minus, l1Bar, model);

                        addProduct(
                            blockIndex, blockIndex, -(2 / std::numbers::pi_v<Real> * Gamma * mu[r]), superfermion[i],
                            superfermion[iBar], returnValue);
                    }
                }
            }
        }

        return returnValue;
    }

    // else: t is not zero

    for (int l1 = 0; l1 < numStates; ++l1)
    {
        for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
        {
            for (int r = 0; r < nRes; ++r)
            {
                Complex Gamma = computeGamma(Eta::Plus, r, l1, l1Bar, model);
                if (Gamma != 0)
                {
                    int i    = singleToMultiIndex(Eta::Plus, l1, model);
                    int iBar = singleToMultiIndex(Eta::Minus, l1Bar, model);
                    if (T[r] == 0)
                    {
                        addProduct(
                            blockIndex, blockIndex, -(2.0 / pi * Gamma * sin(mu[r] * t) / t), superfermion[i],
                            superfermion[iBar], returnValue);
                    }
                    else
                    {
                        addProduct(
                            blockIndex, blockIndex, -(2.0 * Gamma * T[r] * sin(mu[r] * t) / sinh(pi * T[r] * t)),
                            superfermion[i], superfermion[iBar], returnValue);
                    }
                }
            }
        }
    }

    return returnValue;
}

BlockDiagonalMatrix computeGammaGG(SciCore::Real t, const std::vector<BlockMatrix>& superfermion, const Model* model)
{
    using namespace SciCore;

    auto returnValue = BlockDiagonalMatrix::Zero(model->blockDimensions());
    for (int i = 0; i < returnValue.numBlocks(); ++i)
    {
        returnValue(i) = computeGammaGG(i, t, superfermion, model);
    }

    return returnValue;
}

SciCore::RealVector defaultInitialChebSections(SciCore::Real tMax, SciCore::Real tCrit)
{
    using namespace SciCore;

    // We make sure that if a diagram is sampled at 16 Chebyshev points, that the leftmost point is within tCrit.
    // This is to avoid missing narrow features at the beginning.
    Real tSmall = (2 * tCrit) / (1 - std::cos(std::numbers::pi_v<Real> / (2 * 16)));
    if (tSmall <= tMax)
    {
        return RealVector{
            {0, tSmall, tMax}
        };
    }
    else
    {
        return RealVector{
            {0, tMax}
        };
    }
}

SciCore::Real defaultMinChebDistance(SciCore::Real, SciCore::Real)
{
    return 1e-4;
}

} // namespace RealTimeTransport
