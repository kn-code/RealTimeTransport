//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/RenormalizedPT/ConductanceDiagrams.h"

#include <SciCore/Integration.h>

#include "RealTimeTransport/BlockMatrices/MatrixOperations.h"
#include "RealTimeTransport/RenormalizedPT/Diagrams.h"
#include "RealTimeTransport/Utility.h"

namespace RealTimeTransport::RenormalizedPT::Detail
{

//  ______/______
//  |           |
//  |           |
//  0===========0
//
SciCore::Matrix d_dmu_diagram_1(
    int blockIndex,
    SciCore::Real t,
    int r,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::vector<BlockMatrix>& superfermion,
    const Model* model)
{
    using namespace SciCore;

    int numRows              = model->blockDimensions()[blockIndex];
    Matrix returnValue       = Matrix::Zero(numRows, numRows);
    BlockDiagonalMatrix Pi_t = computePi(t);

    int numStates = model->numStates();
    for (int eta = 0; eta <= 1; ++eta)
    {
        for (int l1 = 0; l1 < numStates; ++l1)
        {
            for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
            {
                int i    = singleToMultiIndex(static_cast<Eta>(eta), l1, model);
                int iBar = singleToMultiIndex(static_cast<Eta>(!eta), l1Bar, model);

                Complex prefactor = -d_dmu_gammaMinus(t, static_cast<Eta>(eta), l1, l1Bar, r, model);

                if (prefactor != 0.0)
                {
                    addProduct(
                        blockIndex, blockIndex, prefactor, superfermion[i], Pi_t, superfermion[iBar], returnValue);
                }
            }
        }
    }

    return returnValue;
}

//
//       __
//  ___/__|______
//  |     |     |
//  |     |     |
//  0=====0=====0
//  t     τ     s
//  i2    i1  \bar{i2}
//
// Computes the column col of the above diagram with middle index i1
BlockVector d_dmu_effectiveVertexDiagram1_col(
    int i1,
    int col,
    SciCore::Real t_minus_tau,
    SciCore::Real tau_minus_s,
    int r,
    SciCore::Real epsAbs,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::vector<BlockMatrix>& superfermion,
    const Model* model)
{
    using namespace SciCore;
    using BlockDiagonal = BlockDiagonalMatrix;

    assert(t_minus_tau > 0);
    assert(tau_minus_s > 0);

    const auto& blockDims = model->blockDimensions();
    int numStates         = model->numStates();

    // The return value result represents the column col of the effective vertex i1.
    // Here we assume that if a block of superfermion[i1] is zero, it also follows that
    // the corresponding block result is zero. Might be not true for fancy models (?)
    BlockVector result;
    for (int row = 0; row < superfermion[i1].numBlocks(); ++row)
    {
        auto it = superfermion[i1].find(row, col);
        if (it != superfermion[i1].end())
        {
            const auto& block = it->second;
            result.emplace(row, Matrix::Zero(block.rows(), block.cols()));
        }
    }

    BlockDiagonal Pi1 = (t_minus_tau == 0) ? BlockDiagonal::Identity(blockDims) : computePi(t_minus_tau);
    BlockDiagonal Pi2 = (tau_minus_s == 0) ? BlockDiagonal::Identity(blockDims) : computePi(tau_minus_s);

    BlockMatrix middle = superfermion[i1];
    productCombination_1(Pi1, middle, Pi2);

    for (int eta2 = 0; eta2 <= 1; ++eta2)
    {
        for (int l2 = 0; l2 < numStates; ++l2)
        {
            for (int l2Bar = 0; l2Bar < numStates; ++l2Bar)
            {
                int i2    = singleToMultiIndex(static_cast<Eta>(eta2), l2, model);
                int i2Bar = singleToMultiIndex(static_cast<Eta>(!eta2), l2Bar, model);

                Complex prefactor =
                    d_dmu_gammaMinus(t_minus_tau + tau_minus_s, static_cast<Eta>(eta2), l2, l2Bar, r, model);

                if (prefactor != 0.0)
                {
                    addProduct_col_unsafe(col, prefactor, superfermion[i2], middle, superfermion[i2Bar], result);
                }
            }
        }
    }

    result.eraseZeroes(0.1 * epsAbs);
    return result;
}

//  ______/______
//  |           |
//  |           |
//  0===========O <-- effectiveVertex
//
//  +
//  _____________
//  |           |
//  |           |
//  0===========d/dμ(O) <-- derivative effectiveVertex
//
SciCore::Matrix d_dmu_diagram_2(
    int blockIndex,
    SciCore::Real t,
    int r,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockVector(int, int, SciCore::Real, SciCore::Real)>& computeD_col,
    const std::function<BlockVector(int, int, SciCore::Real, SciCore::Real, int)>& compute_d_dmu_D_col,
    const std::vector<BlockMatrix>& superfermion,
    const Model* model)
{
    assert(t > 0);

    using namespace SciCore;

    int numStates = model->numStates();
    int numRows   = model->blockDimensions()[blockIndex];
    StaticRealVector<2> lower{0, 0};
    StaticRealVector<2> upper{t, 1};

    return integrate2DAdaptive(
        [&](Real s, Real q) -> SciCore::Matrix
        {
            Matrix returnValue = Matrix::Zero(numRows, numRows);

            BlockDiagonalMatrix Pi_t = computePi(s * (1 - q));

            for (int eta = 0; eta <= 1; ++eta)
            {
                for (int l1 = 0; l1 < numStates; ++l1)
                {
                    for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
                    {
                        int i    = singleToMultiIndex(static_cast<Eta>(eta), l1, model);
                        int iBar = singleToMultiIndex(static_cast<Eta>(!eta), l1Bar, model);

                        Complex prefactor1 = -(s * d_dmu_gammaMinus(s, static_cast<Eta>(eta), l1, l1Bar, r, model));
                        Complex prefactor2 = -(s * gammaMinus(s, static_cast<Eta>(eta), l1, l1Bar, model));

                        if (prefactor1 != 0.0)
                        {
                            BlockVector G_iBar = computeD_col(iBar, blockIndex, s * q, t - s);
                            addProduct(blockIndex, prefactor1, superfermion[i], Pi_t, G_iBar, returnValue);
                        }

                        if (prefactor2 != 0.0)
                        {
                            BlockVector d_dmu_G_iBar = compute_d_dmu_D_col(iBar, blockIndex, s * q, t - s, r);
                            addProduct(blockIndex, prefactor2, superfermion[i], Pi_t, d_dmu_G_iBar, returnValue);
                        }
                    }
                }
            }

            return returnValue;
        },
        lower, upper, epsAbs, epsRel);
}

//  ____________/____________
//  |       _________       |
//  |       |       |       |
//  0-------0-------0-------0
//
//  +
//  _________________________
//  |       ____/____       |
//  |       |       |       |
//  0-------0-------0-------0
//
SciCore::Matrix d_dmu_diagram_2_2(
    int blockIndex,
    SciCore::Real t,
    int r,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computeDiagram_1,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& compute_d_dmu_diagram_1,
    const std::vector<BlockMatrix>& superfermion,
    const Model* model)
{
    assert(t > 0);

    using namespace SciCore;

    int numStates = model->numStates();
    int numRows   = model->blockDimensions()[blockIndex];
    StaticRealVector<2> lower{0, 0};
    StaticRealVector<2> upper{1, 1};

    return integrate2DAdaptive(
        [&](Real p, Real q) -> SciCore::Matrix
        {
            Real t1 = p * t;
            Real t2 = q * t1;

            Matrix returnValue = Matrix::Zero(numRows, numRows);

            BlockDiagonalMatrix Pi_1            = computePi(t - t1);
            BlockDiagonalMatrix diagram_1       = computeDiagram_1(t1 - t2);
            BlockDiagonalMatrix d_dmu_diagram_1 = compute_d_dmu_diagram_1(t1 - t2);
            BlockDiagonalMatrix Pi_2            = computePi(t2);

            BlockDiagonalMatrix middle1 = product(1.0, Pi_1, diagram_1, Pi_2);
            BlockDiagonalMatrix middle2 = product(1.0, Pi_1, d_dmu_diagram_1, Pi_2);

            for (int eta = 0; eta <= 1; ++eta)
            {
                for (int l1 = 0; l1 < numStates; ++l1)
                {
                    for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
                    {
                        int i    = singleToMultiIndex(static_cast<Eta>(eta), l1, model);
                        int iBar = singleToMultiIndex(static_cast<Eta>(!eta), l1Bar, model);

                        Complex prefactor1 = -t * t1 * d_dmu_gammaMinus(t, static_cast<Eta>(eta), l1, l1Bar, r, model);
                        Complex prefactor2 = -t * t1 * gammaMinus(t, static_cast<Eta>(eta), l1, l1Bar, model);

                        if (prefactor1 != 0.0)
                        {
                            addProduct(
                                blockIndex, blockIndex, prefactor1, superfermion[i], middle1, superfermion[iBar],
                                returnValue);
                        }
                        if (prefactor2 != 0.0)
                        {
                            addProduct(
                                blockIndex, blockIndex, prefactor2, superfermion[i], middle2, superfermion[iBar],
                                returnValue);
                        }
                    }
                }
            }

            return returnValue;
        },
        lower, upper, epsAbs, epsRel);
}

//  ______/______
//  |           |
//  |           |
//  X===========0
//
Model::SuperRowVectorType d_dmu_currentDiagram_1(
    SciCore::Real t,
    int rI,
    int rmu,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::vector<Model::SuperRowVectorType>& Tr_superfermionAnnihilation,
    const std::vector<BlockMatrix>& superfermion,
    const std::vector<int>& blockStartIndices,
    const Model* model)
{
    assert(t > 0);

    using namespace SciCore;
    using SuperRowVector = Model::SuperRowVectorType;

    int dim       = model->dimHilbertSpace();
    int dim2      = dim * dim;
    int numStates = model->numStates();

    BlockDiagonalMatrix Pi_t = computePi(t);

    SuperRowVector returnValue = SuperRowVector::Zero(1, dim2);

    if (rI != rmu)
    {
        return returnValue;
    }

    for (int eta = 0; eta <= 1; ++eta)
    {
        Real etaEff = (eta == 0) ? -1 : 1;
        for (int l1 = 0; l1 < numStates; ++l1)
        {
            for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
            {
                int i    = singleToMultiIndex(static_cast<Eta>(eta), l1, model);
                int iBar = singleToMultiIndex(static_cast<Eta>(!eta), l1Bar, model);

                Complex prefactor = -(etaEff / 2) * d_dmu_gammaMinus(t, static_cast<Eta>(eta), l1, l1Bar, rmu, model);

                if (prefactor != 0.0)
                {
                    addProduct(
                        prefactor, Tr_superfermionAnnihilation[i], Pi_t, superfermion[iBar], returnValue,
                        blockStartIndices);
                }
            }
        }
    }

    return returnValue;
}

//  ______/______
//  |           |
//  |           |
//  X===========O <-- effectiveVertex
//
//  +
//  _____________
//  |           |
//  |           |
//  X===========d/dμ(O) <-- derivative effectiveVertex
//
Model::SuperRowVectorType d_dmu_currentDiagram_2(
    SciCore::Real t,
    int rI,
    int rmu,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockVector(int, int, SciCore::Real, SciCore::Real)>& computeD_col,
    const std::function<BlockVector(int, int, SciCore::Real, SciCore::Real, int)>& compute_d_dmu_D_col,
    const std::vector<Model::SuperRowVectorType>& Tr_superfermionAnnihilation,
    const std::vector<int>& blockStartIndices,
    int block,
    const Model* model)
{
    assert(t > 0);

    using namespace SciCore;
    using SuperRowVector = Model::SuperRowVectorType;

    int dim       = model->dimHilbertSpace();
    int dim2      = dim * dim;
    int numStates = model->numStates();

    StaticRealVector<2> lower{0, 0};
    StaticRealVector<2> upper{t, 1};

    auto d_dmu_D_col = [&](int i, int j, Real t1, Real t2)
    {
        return compute_d_dmu_D_col(i, j, t1, t2, rmu);
    };

    return integrate2DAdaptive(
        [&](Real s, Real q) -> SuperRowVector
        {
            SuperRowVector returnValue = SuperRowVector::Zero(1, dim2);

            BlockDiagonalMatrix Pi_t = computePi(s * (1 - q));

            for (int eta = 0; eta <= 1; ++eta)
            {
                Real etaEff = (eta == 0) ? -1 : 1;
                for (int l1 = 0; l1 < numStates; ++l1)
                {
                    for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
                    {
                        int i    = singleToMultiIndex(static_cast<Eta>(eta), l1, model);
                        int iBar = singleToMultiIndex(static_cast<Eta>(!eta), l1Bar, model);

                        Complex prefactor1 =
                            (rmu != rI) ? 0.0
                                        : (-(etaEff / 2) *
                                           (s * d_dmu_gammaMinus(s, static_cast<Eta>(eta), l1, l1Bar, rmu, model)));
                        Complex prefactor2 =
                            -(etaEff / 2) * (s * gammaMinus(s, static_cast<Eta>(eta), l1, l1Bar, rI, model));

                        if (prefactor1 != 0.0)
                        {
                            BlockMatrix D_iBar =
                                effectiveVertexDiagram1_fromCols(iBar, s * q, t - s, computeD_col, block, model);
                            addProduct(
                                prefactor1, Tr_superfermionAnnihilation[i], Pi_t, D_iBar, returnValue,
                                blockStartIndices);
                        }

                        if (prefactor2 != 0.0)
                        {
                            BlockMatrix d_dmu_D_iBar =
                                effectiveVertexDiagram1_fromCols(iBar, s * q, t - s, d_dmu_D_col, block, model);
                            addProduct(
                                prefactor2, Tr_superfermionAnnihilation[i], Pi_t, d_dmu_D_iBar, returnValue,
                                blockStartIndices);
                        }
                    }
                }
            }

            return returnValue;
        },
        lower, upper, epsAbs, epsRel);
}

//  ____________/____________
//  |       _________       |
//  |       |       |       |
//  X-------0-------0-------0
//
//  +
//  _________________________
//  |       ____/____       |
//  |       |       |       |
//  X-------0-------0-------0
//
Model::SuperRowVectorType d_dmu_currentDiagram_2_2(
    SciCore::Real t,
    int rI,
    int rmu,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computeDiagram_1,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& compute_d_dmu_diagram_1,
    const std::vector<BlockMatrix>& superfermion,
    const std::vector<Model::SuperRowVectorType>& Tr_superfermionAnnihilation,
    const std::vector<int>& blockStartIndices,
    const Model* model)
{
    assert(t > 0);

    using namespace SciCore;
    using SuperRowVector = Model::SuperRowVectorType;

    int dim       = model->dimHilbertSpace();
    int dim2      = dim * dim;
    int numStates = model->numStates();

    StaticRealVector<2> lower{0, 0};
    StaticRealVector<2> upper{1, 1};

    return integrate2DAdaptive(
        [&](Real p, Real q) -> SuperRowVector
        {
            Real t1 = p * t;
            Real t2 = q * t1;

            SuperRowVector returnValue = SuperRowVector::Zero(1, dim2);

            BlockDiagonalMatrix Pi_1            = computePi(t - t1);
            BlockDiagonalMatrix diagram_1       = computeDiagram_1(t1 - t2);
            BlockDiagonalMatrix d_dmu_diagram_1 = compute_d_dmu_diagram_1(t1 - t2);
            BlockDiagonalMatrix Pi_2            = computePi(t2);

            BlockDiagonalMatrix middle1 = product(1.0, Pi_1, diagram_1, Pi_2);
            BlockDiagonalMatrix middle2 = product(1.0, Pi_1, d_dmu_diagram_1, Pi_2);

            for (int eta = 0; eta <= 1; ++eta)
            {
                Real etaEff = (eta == 0) ? -1 : 1;
                for (int l1 = 0; l1 < numStates; ++l1)
                {
                    for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
                    {
                        int i    = singleToMultiIndex(static_cast<Eta>(eta), l1, model);
                        int iBar = singleToMultiIndex(static_cast<Eta>(!eta), l1Bar, model);

                        Complex prefactor1 =
                            (rmu != rI) ? 0.0
                                        : (-(etaEff / 2) * t *
                                           (t1 * d_dmu_gammaMinus(t, static_cast<Eta>(eta), l1, l1Bar, rmu, model)));

                        Complex prefactor2 =
                            -(etaEff / 2) * t * (t1 * gammaMinus(t, static_cast<Eta>(eta), l1, l1Bar, rI, model));

                        if (prefactor1 != 0.0)
                        {
                            addProduct(
                                prefactor1, Tr_superfermionAnnihilation[i], middle1, superfermion[iBar], returnValue,
                                blockStartIndices);
                        }

                        if (prefactor2 != 0.0)
                        {
                            addProduct(
                                prefactor2, Tr_superfermionAnnihilation[i], middle2, superfermion[iBar], returnValue,
                                blockStartIndices);
                        }
                    }
                }
            }

            return returnValue;
        },
        lower, upper, epsAbs, epsRel);
}

} // namespace RealTimeTransport::RenormalizedPT::Detail
