//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include <SciCore/Integration.h>

#include "RealTimeTransport/BlockMatrices/MatrixOperations.h"
#include "RealTimeTransport/IteratedRG/Diagrams.h"
#include "RealTimeTransport/RenormalizedPT/Diagrams.h"
#include "RealTimeTransport/Utility.h"

namespace RealTimeTransport::IteratedRG::Detail
{

//
//       __    __
//  ______|_____|______
//  |     |     |     |
//  |     |     |     |
//  0=====0=====0=====0
//  t    τ_1   τ_2    s
//  i3    i1    i2  \bar{i3}
//
BlockVector<SciCore::Complex> bareTwoPointVertex_col(
    int i1,
    int i2,
    int col,
    SciCore::Real t_minus_tau1,
    SciCore::Real tau1_minus_tau2,
    SciCore::Real tau2_minus_s,
    const Model::BlockDiagonalType& Pi1,
    const Model::BlockDiagonalType& Pi2,
    const Model::BlockDiagonalType& Pi3,
    SciCore::Real epsAbs,
    const std::vector<Model::SuperfermionType>& superfermion,
    const Model* model)
{
    using namespace SciCore;

    assert(t_minus_tau1 >= 0);
    assert(tau1_minus_tau2 >= 0);
    assert(tau2_minus_s >= 0);

    BlockVector<Complex> result;

    int numStates = model->numStates();

    // dressed_i1 = Pi1 * superfermion[i1] * Pi2
    BlockMatrix<Complex> dressed_i1 = superfermion[i1];
    productCombination_1(Pi1, dressed_i1, Pi2);

    for (int eta3 = 0; eta3 <= 1; ++eta3)
    {
        for (int l3 = 0; l3 < numStates; ++l3)
        {
            for (int l3Bar = 0; l3Bar < numStates; ++l3Bar)
            {
                int i3    = singleToMultiIndex(static_cast<Eta>(eta3), l3, model);
                int i3Bar = singleToMultiIndex(static_cast<Eta>(!eta3), l3Bar, model);

                Complex prefactor = -gammaMinus(
                    t_minus_tau1 + tau1_minus_tau2 + tau2_minus_s, static_cast<Eta>(eta3), l3, l3Bar, model);

                if (prefactor != 0.0)
                {
                    Model::SuperfermionType left = product(prefactor, superfermion[i3], dressed_i1, superfermion[i2]);
                    addProduct_col(col, 1.0, left, Pi3, superfermion[i3Bar], result);
                }
            }
        }
    }

    result.eraseZeroes(0.1 * epsAbs);
    return result;
}

//
//       __
//  ______|______
//  |     |     |
//  |     |     |
//  0=====0=====E  <-- effective vertex O(D^3)
//  t     τ
//  i2    i1  \bar{i2}
//
// Computes the column col of the above diagram with middle index i1
BlockVector<SciCore::Complex> effectiveVertexCorrection1_col(
    int i1,
    int col,
    SciCore::Real t_minus_tau,
    SciCore::Real tau,
    SciCore::Real epsAbs,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePi,
    const std::function<BlockVector<SciCore::Complex>(int, int, SciCore::Real, SciCore::Real)>& computeD_col,
    const std::vector<Model::SuperfermionType>& superfermion,
    const Model* model)
{
    using namespace SciCore;
    using BlockDiagonal = Model::BlockDiagonalType;
    using MatrixType    = BlockVector<Complex>::MatrixType;

    assert(t_minus_tau > 0);
    assert(tau > 0);

    // The return value represents the column col of the above diagram with middle index i1.
    // Here we assume that if a block of superfermion[i1] is zero, it also follows that
    // the corresponding block of the effective vertex i1 is zero. Might be not true for fancy models (?)
    BlockVector<Complex> result;
    for (int row = 0; row < superfermion[i1].numBlocks(); ++row)
    {
        auto it = superfermion[i1].find(row, col);
        if (it != superfermion[i1].end())
        {
            const auto& block = it->second;
            result.emplace(row, Matrix::Zero(block.rows(), block.cols()));
        }
    }

    Real epsRel   = 0;
    int numStates = model->numStates();

    StaticRealVector<2> lower{0, 0};
    StaticRealVector<2> upper{1, 1};
    for (auto& element : result)
    {
        int rowIndex          = element.first;
        MatrixType& resultRow = element.second;
        MatrixType integrand(resultRow.rows(), resultRow.cols());

        resultRow = integrate2DAdaptive(
            [&](Real p, Real q) -> MatrixType
            {
                integrand.setZero();

                Real t  = t_minus_tau + tau;
                Real t2 = p * tau;
                Real t1 = q * (tau - t2) + t2;

                BlockDiagonal Pi_t_minus_tau  = computePi(t_minus_tau);
                BlockDiagonal Pi_tau_minus_t1 = computePi(tau - t1);

                BlockMatrix<Complex> middle = superfermion[i1];
                productCombination_1(Pi_t_minus_tau, middle, Pi_tau_minus_t1);

                for (int eta2 = 0; eta2 <= 1; ++eta2)
                {
                    for (int l2 = 0; l2 < numStates; ++l2)
                    {
                        for (int l2Bar = 0; l2Bar < numStates; ++l2Bar)
                        {
                            int i2    = singleToMultiIndex(static_cast<Eta>(eta2), l2, model);
                            int i2Bar = singleToMultiIndex(static_cast<Eta>(!eta2), l2Bar, model);

                            Complex prefactor =
                                tau * ((tau - t2) * gammaMinus(t - t2, static_cast<Eta>(eta2), l2, l2Bar, model));

                            if (prefactor != 0.0)
                            {
                                BlockVector<SciCore::Complex> Geff_i2Bar = computeD_col(i2Bar, col, t1 - t2, t2);
                                addProduct(rowIndex, prefactor, superfermion[i2], middle, Geff_i2Bar, integrand);
                            }
                        }
                    }
                }

                return integrand;
            },
            lower, upper, epsAbs, epsRel);
    }

    result.eraseZeroes(0.1 * epsAbs);
    return result;
}

//
//       __
//  ______|______
//  |     |     |
//  |     |     |
//  0=====E=====0  ( E is effective vertex O(D^3) )
//  t     τ
//  i2    i1  \bar{i2}
//
// Computes the column col of the above diagram with middle index i1
BlockVector<SciCore::Complex> effectiveVertexCorrection2_col(
    int i1,
    int col,
    SciCore::Real t_minus_tau,
    SciCore::Real tau,
    SciCore::Real epsAbs,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePi,
    const std::function<BlockVector<SciCore::Complex>(int, int, SciCore::Real, SciCore::Real)>& computeD_col,
    const std::vector<Model::SuperfermionType>& superfermion,
    const Model* model)
{
    using namespace SciCore;
    using BlockDiagonal = Model::BlockDiagonalType;
    using MatrixType    = BlockVector<Complex>::MatrixType;

    assert(t_minus_tau > 0);
    assert(tau > 0);

    // The return value represents the column col of the above diagram with middle index i1.
    // Here we assume that if a block of superfermion[i1] is zero, it also follows that
    // the corresponding block of the effective vertex i1 is zero. Might be not true for fancy models (?)
    BlockVector<Complex> result;
    for (int row = 0; row < superfermion[i1].numBlocks(); ++row)
    {
        auto it = superfermion[i1].find(row, col);
        if (it != superfermion[i1].end())
        {
            const auto& block = it->second;
            result.emplace(row, Matrix::Zero(block.rows(), block.cols()));
        }
    }

    Real epsRel   = 0;
    int numStates = model->numStates();

    StaticRealVector<2> lower{0, 0};
    StaticRealVector<2> upper{1, 1};
    for (auto& element : result)
    {
        int rowIndex          = element.first;
        MatrixType& resultRow = element.second;
        MatrixType integrand(resultRow.rows(), resultRow.cols());

        resultRow = integrate2DAdaptive(
            [&](Real p, Real q) -> MatrixType
            {
                integrand.setZero();

                Real t  = t_minus_tau + tau;
                Real t2 = q * tau;
                Real t1 = p * t_minus_tau + tau;

                BlockDiagonal Pi_t_minus_t1 = computePi(t - t1);
                BlockDiagonal Pi_t2         = computePi(t2);

                Model::SuperfermionType Geff_i1 = RenormalizedPT::Detail::effectiveVertexDiagram1_fromCols(
                    i1, t1 - tau, tau - t2, computeD_col, -1, model);
                productCombination_1(Pi_t_minus_t1, Geff_i1, Pi_t2);

                for (int eta2 = 0; eta2 <= 1; ++eta2)
                {
                    for (int l2 = 0; l2 < numStates; ++l2)
                    {
                        for (int l2Bar = 0; l2Bar < numStates; ++l2Bar)
                        {
                            int i2    = singleToMultiIndex(static_cast<Eta>(eta2), l2, model);
                            int i2Bar = singleToMultiIndex(static_cast<Eta>(!eta2), l2Bar, model);

                            Complex prefactor =
                                t_minus_tau * (tau * gammaMinus(t, static_cast<Eta>(eta2), l2, l2Bar, model));

                            if (prefactor != 0.0)
                            {
                                addProduct(
                                    rowIndex, col, prefactor, superfermion[i2], Geff_i1, superfermion[i2Bar],
                                    integrand);
                            }
                        }
                    }
                }

                return integrand;
            },
            lower, upper, epsAbs, epsRel);
    }

    result.eraseZeroes(0.1 * epsAbs);
    return result;
}

//
//              __
//  _____________|
//  |           ||
//  |           ||
//  0===========00
//  t            τ
//  i2       \bar{i2},i1
//
BlockVector<SciCore::Complex> effectiveVertexCorrection3_col(
    int i1,
    int col,
    SciCore::Real t_minus_tau,
    SciCore::Real tau,
    SciCore::Real epsAbs,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePi,
    const std::vector<Model::SuperfermionType>& superfermion,
    const Model* model)
{
    using namespace SciCore;
    using BlockDiagonal = Model::BlockDiagonalType;
    using MatrixType    = BlockVector<Complex>::MatrixType;

    assert(t_minus_tau > 0);
    assert(tau > 0);

    // The return value represents the column col of the above diagram with free index i1.
    // Here we assume that if a block of superfermion[i1] is zero, it also follows that
    // the corresponding block of the effective vertex i1 is zero. Might be not true for fancy models (?)
    BlockVector<Complex> result;
    for (int row = 0; row < superfermion[i1].numBlocks(); ++row)
    {
        auto it = superfermion[i1].find(row, col);
        if (it != superfermion[i1].end())
        {
            const auto& block = it->second;
            result.emplace(row, Matrix::Zero(block.rows(), block.cols()));
        }
    }

    Real epsRel   = 0;
    int numStates = model->numStates();

    StaticRealVector<2> lower{0, 0};
    StaticRealVector<2> upper{1, 1};

    for (auto& element : result)
    {
        int rowIndex          = element.first;
        MatrixType& resultRow = element.second;
        MatrixType integrand(resultRow.rows(), resultRow.cols());

        BlockDiagonal Pi_tau = computePi(tau);

        resultRow = integrate2DAdaptive(
            [&](Real p, Real q) -> MatrixType
            {
                integrand.setZero();

                Real t  = t_minus_tau + tau;
                Real t2 = p * t_minus_tau + tau;
                Real t1 = q * (t - t2) + t2;

                BlockDiagonal Pi              = computePi(t - t1);
                BlockDiagonal Pi_t1_minus_t2  = computePi(t1 - t2);
                BlockDiagonal Pi_t2_minus_tau = computePi(t2 - tau);

                for (int eta2 = 0; eta2 <= 1; ++eta2)
                {
                    for (int l2 = 0; l2 < numStates; ++l2)
                    {
                        for (int l2Bar = 0; l2Bar < numStates; ++l2Bar)
                        {
                            int i2    = singleToMultiIndex(static_cast<Eta>(eta2), l2, model);
                            int i2Bar = singleToMultiIndex(static_cast<Eta>(!eta2), l2Bar, model);

                            Complex prefactor =
                                -t_minus_tau *
                                ((t - t2) * gammaMinus(t - t2, static_cast<Eta>(eta2), l2, l2Bar, model));

                            if (prefactor != 0.0)
                            {
                                BlockVector<SciCore::Complex> D2_col = bareTwoPointVertex_col(
                                    i2Bar, i1, col, t1 - t2, t2 - tau, tau, Pi_t1_minus_t2, Pi_t2_minus_tau, Pi_tau,
                                    epsAbs, superfermion, model);

                                addProduct(rowIndex, prefactor, superfermion[i2], Pi, D2_col, integrand);
                            }
                        }
                    }
                }

                return integrand;
            },
            lower, upper, epsAbs, epsRel);
    }

    result.eraseZeroes(0.1 * epsAbs);
    return result;
}

//
//             __
//  ____________|_
//  |           ||
//  |           ||
//  0===========00
//  t           τ
//  i2        i1,\bar{i2}
//
BlockVector<SciCore::Complex> effectiveVertexCorrection4_col(
    int i1,
    int col,
    SciCore::Real t_minus_tau,
    SciCore::Real tau,
    SciCore::Real epsAbs,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePi,
    const std::vector<Model::SuperfermionType>& superfermion,
    const Model* model)
{
    using namespace SciCore;
    using BlockDiagonal = Model::BlockDiagonalType;
    using MatrixType    = BlockVector<Complex>::MatrixType;

    assert(t_minus_tau > 0);
    assert(tau > 0);

    // The return value represents the column col of the above diagram with free index i1.
    // Here we assume that if a block of superfermion[i1] is zero, it also follows that
    // the corresponding block of the effective vertex i1 is zero. Might be not true for fancy models (?)
    BlockVector<Complex> result;
    for (int row = 0; row < superfermion[i1].numBlocks(); ++row)
    {
        auto it = superfermion[i1].find(row, col);
        if (it != superfermion[i1].end())
        {
            const auto& block = it->second;
            result.emplace(row, Matrix::Zero(block.rows(), block.cols()));
        }
    }

    Real epsRel   = 0;
    int numStates = model->numStates();

    StaticRealVector<2> lower{0, 0};
    StaticRealVector<2> upper{1, 1};

    for (auto& element : result)
    {
        int rowIndex          = element.first;
        MatrixType& resultRow = element.second;
        MatrixType integrand(resultRow.rows(), resultRow.cols());

        resultRow = integrate2DAdaptive(
            [&](Real p, Real q) -> MatrixType
            {
                integrand.setZero();

                Real t  = t_minus_tau + tau;
                Real t2 = q * tau;
                Real t1 = p * t_minus_tau + tau;

                BlockDiagonal Pi              = computePi(t - t1);
                BlockDiagonal Pi_t1_minus_tau = computePi(t1 - tau);
                BlockDiagonal Pi_tau_minus_t2 = computePi(tau - t2);
                BlockDiagonal Pi_t2           = computePi(t2);

                for (int eta2 = 0; eta2 <= 1; ++eta2)
                {
                    for (int l2 = 0; l2 < numStates; ++l2)
                    {
                        for (int l2Bar = 0; l2Bar < numStates; ++l2Bar)
                        {
                            int i2    = singleToMultiIndex(static_cast<Eta>(eta2), l2, model);
                            int i2Bar = singleToMultiIndex(static_cast<Eta>(!eta2), l2Bar, model);

                            Complex prefactor =
                                tau * (t_minus_tau * gammaMinus(t - t2, static_cast<Eta>(eta2), l2, l2Bar, model));

                            if (prefactor != 0.0)
                            {
                                BlockVector<SciCore::Complex> D2_col = bareTwoPointVertex_col(
                                    i1, i2Bar, col, t1 - tau, tau - t2, t2, Pi_t1_minus_tau, Pi_tau_minus_t2, Pi_t2,
                                    epsAbs, superfermion, model);

                                addProduct(rowIndex, prefactor, superfermion[i2], Pi, D2_col, integrand);
                            }
                        }
                    }
                }

                return integrand;
            },
            lower, upper, epsAbs, epsRel);
    }

    result.eraseZeroes(0.1 * epsAbs);
    return result;
}

} // namespace RealTimeTransport::IteratedRG::Detail
