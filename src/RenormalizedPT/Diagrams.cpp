//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include <SciCore/Integration.h>

#include "RealTimeTransport/BlockMatrices/MatrixOperations.h"
#include "RealTimeTransport/RenormalizedPT/Diagrams.h"

namespace RealTimeTransport::RenormalizedPT::Detail
{

SciCore::Matrix diagram_1_regular(
    int blockIndex,
    SciCore::Real t,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePi,
    const std::vector<Model::SuperfermionType>& superfermion,
    const Model* model)
{
    using namespace SciCore;

    int numRows                   = model->blockDimensions()[blockIndex];
    Matrix returnValue            = Matrix::Zero(numRows, numRows);
    Model::BlockDiagonalType Pi_t = computePi(t);

    int numStates = model->numStates();
    for (int eta = 0; eta <= 1; ++eta)
    {
        for (int l1 = 0; l1 < numStates; ++l1)
        {
            for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
            {
                int i    = singleToMultiIndex(static_cast<Eta>(eta), l1, model);
                int iBar = singleToMultiIndex(static_cast<Eta>(!eta), l1Bar, model);

                Complex prefactor = -gammaMinus(t, static_cast<Eta>(eta), l1, l1Bar, model);

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

SciCore::Matrix diagram_1_small_t(
    int blockIndex,
    SciCore::Real t,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePiMinusOne,
    const std::vector<Model::SuperfermionType>& superfermion,
    const Model* model)
{
    assert(t > 0);

    using namespace SciCore;

    int numRows                           = model->blockDimensions()[blockIndex];
    Matrix returnValue                    = Matrix::Zero(numRows, numRows);
    Model::BlockDiagonalType PiMinusOne_t = computePiMinusOne(t);

    int numStates = model->numStates();

    for (int eta = 0; eta <= 1; ++eta)
    {
        for (int l1 = 0; l1 < numStates; ++l1)
        {
            for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
            {
                int i    = singleToMultiIndex(static_cast<Eta>(eta), l1, model);
                int iBar = singleToMultiIndex(static_cast<Eta>(!eta), l1Bar, model);

                Complex prefactor = -gammaMinus(t, static_cast<Eta>(eta), l1, l1Bar, model);

                if (prefactor != 0.0)
                {
                    addProduct(
                        blockIndex, blockIndex, prefactor, superfermion[i], PiMinusOne_t, superfermion[iBar],
                        returnValue);
                }
            }
        }
    }

    returnValue -= computeGammaGG(blockIndex, t, superfermion, model);
    return returnValue;
}

//  _____________
//  |           |
//  |           |
//  0===========0
//
SciCore::Matrix diagram_1(
    int blockIndex,
    SciCore::Real t,
    SciCore::Real tCrit,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePi,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePiMinusOne,
    const std::vector<Model::SuperfermionType>& superfermion,
    const Model* model)
{
    if (t > tCrit)
    {
        return diagram_1_regular(blockIndex, t, computePi, superfermion, model);
    }
    else
    {
        return diagram_1_small_t(blockIndex, t, computePiMinusOne, superfermion, model);
    }
}

//
//       __
//  ______|______
//  |     |     |
//  |     |     |
//  0=====0=====0
//  t     τ     s
//  i2    i1  \bar{i2}
//
// Computes the column col of the above diagram with middle index i1
BlockVector<SciCore::Complex> effectiveVertexDiagram1_col(
    int i1,
    int col,
    SciCore::Real t_minus_tau,
    SciCore::Real tau_minus_s,
    SciCore::Real tCrit,
    SciCore::Real epsAbs,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePi,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePiMinusOne,
    const std::vector<Model::SuperfermionType>& superfermion,
    const Model* model)
{
    using namespace SciCore;
    using BlockDiagonal = Model::BlockDiagonalType;

    assert(t_minus_tau > 0);
    assert(tau_minus_s > 0);

    const auto& blockDims = model->blockDimensions();
    int numStates         = model->numStates();

    // The return value result represents the column col of the effective vertex i1.
    // Here we assume that if a block of superfermion[i1] is zero, it also follows that
    // the corresponding block result is zero. Might be not true for fancy models (?)
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

    //if (t_minus_tau + tau_minus_s >= tCrit)
    //{
    BlockDiagonal Pi1 = (t_minus_tau == 0) ? BlockDiagonal::Identity(blockDims) : computePi(t_minus_tau);
    BlockDiagonal Pi2 = (tau_minus_s == 0) ? BlockDiagonal::Identity(blockDims) : computePi(tau_minus_s);

    BlockMatrix<Complex> middle = superfermion[i1];
    productCombination_1(Pi1, middle, Pi2);

    for (int eta2 = 0; eta2 <= 1; ++eta2)
    {
        for (int l2 = 0; l2 < numStates; ++l2)
        {
            for (int l2Bar = 0; l2Bar < numStates; ++l2Bar)
            {
                int i2    = singleToMultiIndex(static_cast<Eta>(eta2), l2, model);
                int i2Bar = singleToMultiIndex(static_cast<Eta>(!eta2), l2Bar, model);

                Complex prefactor = gammaMinus(t_minus_tau + tau_minus_s, static_cast<Eta>(eta2), l2, l2Bar, model);

                if (prefactor != 0.0)
                {
                    addProduct_col_unsafe(col, prefactor, superfermion[i2], middle, superfermion[i2Bar], result);
                }
            }
        }
    }
    //}
    /*else
    {
        BlockDiagonal PiMinusOne1 =
            (t_minus_tau == 0) ? BlockDiagonal::Zero(blockDims) : computePiMinusOne(t_minus_tau);
        BlockDiagonal PiMinusOne2 =
            (tau_minus_s == 0) ? BlockDiagonal::Zero(blockDims) : computePiMinusOne(tau_minus_s);
        BlockDiagonal gammaGG = computeGammaGG(t_minus_tau + tau_minus_s, superfermion, model);

        BlockMatrix<Complex> middle = superfermion[i1];
        productCombination_2(PiMinusOne1, middle, PiMinusOne2);

        for (int eta2 = 0; eta2 <= 1; ++eta2)
        {
            for (int l2 = 0; l2 < numStates; ++l2)
            {
                for (int l2Bar = 0; l2Bar < numStates; ++l2Bar)
                {
                    int i2    = singleToMultiIndex(static_cast<Eta>(eta2), l2, model);
                    int i2Bar = singleToMultiIndex(static_cast<Eta>(!eta2), l2Bar, model);

                    Complex prefactor = gammaMinus(t_minus_tau + tau_minus_s, static_cast<Eta>(eta2), l2, l2Bar, model);

                    if (prefactor != 0.0)
                    {
                        addProduct_col_unsafe(col, prefactor, superfermion[i2], middle, superfermion[i2Bar], result);
                    }
                }
            }
        }

        addProduct_col_unsafe_2(col, -1.0, superfermion[i1], gammaGG, result);
    }
    */

    result.eraseZeroes(0.1 * epsAbs);
    return result;
}

Model::SuperfermionType effectiveVertexDiagram1_fromCols(
    int i1,
    SciCore::Real t_minus_tau,
    SciCore::Real tau_minus_s,
    const std::function<BlockVector<SciCore::Complex>(int, int, SciCore::Real, SciCore::Real)>& computeD_col,
    int block,
    const Model* model)
{
    using namespace SciCore;
    using UnorderedElementMap = Model::SuperfermionType::UnorderedElementMap;

    assert(t_minus_tau > 0);
    assert(tau_minus_s > 0);

    const auto& blockDims = model->blockDimensions();
    int numBlocks         = static_cast<int>(blockDims.size());

    UnorderedElementMap resultAsMap;

    if (block < 0)
    {
        // First compute column 0, then column 1, ...
        for (int col = 0; col < numBlocks; ++col)
        {
            BlockVector<SciCore::Complex> resultCol = computeD_col(i1, col, t_minus_tau, tau_minus_s);

            for (auto& element : resultCol)
            {
                int row                 = element.first;
                resultAsMap[{row, col}] = std::move(element.second);
            }
        }
    }
    else
    {
        int col = block;

        BlockVector<SciCore::Complex> resultCol = computeD_col(i1, col, t_minus_tau, tau_minus_s);

        for (auto& element : resultCol)
        {
            int row                 = element.first;
            resultAsMap[{row, col}] = std::move(element.second);
        }
    }

    return Model::SuperfermionType(std::move(resultAsMap), blockDims);
}

//  _____________
//  |           |
//  |           |
//  0===========O <-- effectiveVertex
//
SciCore::Matrix diagram_2(
    int blockIndex,
    SciCore::Real t,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePi,
    const std::function<BlockVector<SciCore::Complex>(int, int, SciCore::Real, SciCore::Real)>& computeD_col,
    const std::vector<Model::SuperfermionType>& superfermion,
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

            Model::BlockDiagonalType Pi_t = computePi(s * (1 - q));

            for (int eta = 0; eta <= 1; ++eta)
            {
                for (int l1 = 0; l1 < numStates; ++l1)
                {
                    for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
                    {
                        int i    = singleToMultiIndex(static_cast<Eta>(eta), l1, model);
                        int iBar = singleToMultiIndex(static_cast<Eta>(!eta), l1Bar, model);

                        Complex prefactor = -(s * gammaMinus(s, static_cast<Eta>(eta), l1, l1Bar, model));

                        if (prefactor != 0.0)
                        {
                            BlockVector<SciCore::Complex> G_iBar = computeD_col(iBar, blockIndex, s * q, t - s);
                            addProduct(blockIndex, prefactor, superfermion[i], Pi_t, G_iBar, returnValue);
                        }
                    }
                }
            }

            return returnValue;
        },
        lower, upper, epsAbs, epsRel);
}

//  _________________________
//  |       _________       |
//  |       |       |       |
//  0-------0-------0-------0
//
SciCore::Matrix diagram_2_2(
    int blockIndex,
    SciCore::Real t,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePi,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computeDiagram_1,
    const std::vector<Model::SuperfermionType>& superfermion,
    const Model* model)
{
    assert(t > 0);

    using namespace SciCore;

    int numStates = model->numStates();
    int numRows   = model->blockDimensions()[blockIndex];
    StaticRealVector<2> lower{0, 0};
    StaticRealVector<2> upper{1, 1};

    // It appears that this safety factor is sometimes necessary.
    Real safety = 0.1;
    return integrate2DAdaptive(
        [&](Real p, Real q) -> SciCore::Matrix
        {
            Real t1 = p * t;
            Real t2 = q * t1;

            Matrix returnValue = Matrix::Zero(numRows, numRows);

            Model::BlockDiagonalType Pi_1      = computePi(t - t1);
            Model::BlockDiagonalType diagram_1 = computeDiagram_1(t1 - t2);
            Model::BlockDiagonalType Pi_2      = computePi(t2);

            Model::BlockDiagonalType middle = product(1.0, Pi_1, diagram_1, Pi_2);

            for (int eta = 0; eta <= 1; ++eta)
            {
                for (int l1 = 0; l1 < numStates; ++l1)
                {
                    for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
                    {
                        int i    = singleToMultiIndex(static_cast<Eta>(eta), l1, model);
                        int iBar = singleToMultiIndex(static_cast<Eta>(!eta), l1Bar, model);

                        Complex prefactor = -t * t1 * gammaMinus(t, static_cast<Eta>(eta), l1, l1Bar, model);

                        if (prefactor != 0.0)
                        {
                            addProduct(
                                blockIndex, blockIndex, prefactor, superfermion[i], middle, superfermion[iBar],
                                returnValue);
                        }
                    }
                }
            }

            return returnValue;
        },
        lower, upper, safety* epsAbs, safety* epsRel);
}

//  _____________
//  |           |
//  |           |
//  X===========0
//
Model::SuperRowVectorType currentDiagram_1_regular(
    SciCore::Real t,
    int r,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePi,
    const std::vector<Model::SuperRowVectorType>& Tr_superfermionAnnihilation,
    const std::vector<Model::SuperfermionType>& superfermion,
    const std::vector<int>& blockStartIndices,
    const Model* model)
{
    assert(t > 0);

    using namespace SciCore;
    using SuperRowVector = Model::SuperRowVectorType;

    int dim       = model->dimHilbertSpace();
    int dim2      = dim * dim;
    int numStates = model->numStates();

    Model::BlockDiagonalType Pi_t = computePi(t);

    SuperRowVector returnValue = SuperRowVector::Zero(1, dim2);
    for (int eta = 0; eta <= 1; ++eta)
    {
        Real etaEff = (eta == 0) ? -1 : 1;
        for (int l1 = 0; l1 < numStates; ++l1)
        {
            for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
            {
                int i    = singleToMultiIndex(static_cast<Eta>(eta), l1, model);
                int iBar = singleToMultiIndex(static_cast<Eta>(!eta), l1Bar, model);

                Complex prefactor = -(etaEff / 2) * gammaMinus(t, static_cast<Eta>(eta), l1, l1Bar, r, model);

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

//  _____________
//  |           |
//  |           |
//  X===========0
//
Model::SuperRowVectorType currentDiagram_1_small_t(
    SciCore::Real t,
    int r,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePiMinusOne,
    const Model::SuperRowVectorType& idRow,
    const std::vector<Model::SuperRowVectorType>& Tr_superfermionAnnihilation,
    const std::vector<Model::SuperfermionType>& superfermion,
    const std::vector<int>& blockStartIndices,
    const Model* model)
{
    assert(t > 0);

    using namespace SciCore;
    using SuperRowVector = Model::SuperRowVectorType;

    int dim       = model->dimHilbertSpace();
    int dim2      = dim * dim;
    int numStates = model->numStates();

    Model::BlockDiagonalType PiMinusOne = computePiMinusOne(t);

    SuperRowVector returnValue = SuperRowVector::Zero(1, dim2);
    for (int eta = 0; eta <= 1; ++eta)
    {
        Real etaEff = (eta == 0) ? -1 : 1;
        for (int l1 = 0; l1 < numStates; ++l1)
        {
            for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
            {
                int i    = singleToMultiIndex(static_cast<Eta>(eta), l1, model);
                int iBar = singleToMultiIndex(static_cast<Eta>(!eta), l1Bar, model);

                Complex prefactor = -(etaEff / 2) * gammaMinus(t, static_cast<Eta>(eta), l1, l1Bar, r, model);

                if (prefactor != 0.0)
                {
                    addProduct(
                        prefactor, Tr_superfermionAnnihilation[i], PiMinusOne, superfermion[iBar], returnValue,
                        blockStartIndices);
                }
            }
        }
    }

    const RealVector& temperatures       = model->temperatures();
    const RealVector& chemicalPotentials = model->chemicalPotentials();

    Complex term2(0, 0);
    for (int l = 0; l < numStates; ++l)
    {
        term2 += computeGamma(Eta::Plus, r, l, l, model);
    }

    if (t != 0)
    {
        if (temperatures[r] != 0)
        {
            term2 *= temperatures[r] * std::sin(chemicalPotentials[r] * t) /
                     std::sinh(std::numbers::pi_v<SciCore::Real> * temperatures[r] * t);
        }
        else
        {
            term2 *= std::sin(chemicalPotentials[r] * t) / (std::numbers::pi_v<SciCore::Real> * t);
        }
    }
    else
    {
        term2 *= chemicalPotentials[r] / std::numbers::pi_v<SciCore::Real>;
    }

    returnValue.noalias() += term2 * idRow;

    return returnValue;
}

//  _____________
//  |           |
//  |           |
//  X===========0
//
Model::SuperRowVectorType currentDiagram_1(
    SciCore::Real t,
    int r,
    SciCore::Real tCrit,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePi,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePiMinusOne,
    const Model::SuperRowVectorType& idRow,
    const std::vector<Model::SuperRowVectorType>& Tr_superfermionAnnihilation,
    const std::vector<Model::SuperfermionType>& superfermion,
    const std::vector<int>& blockStartIndices,
    const Model* model)
{
    if (t > tCrit)
    {
        return currentDiagram_1_regular(
            t, r, computePi, Tr_superfermionAnnihilation, superfermion, blockStartIndices, model);
    }
    else
    {
        return currentDiagram_1_small_t(
            t, r, computePiMinusOne, idRow, Tr_superfermionAnnihilation, superfermion, blockStartIndices, model);
    }
}

//  _____________
//  |           |
//  |           |
//  X===========O <-- effectiveVertex
//
Model::SuperRowVectorType currentDiagram_2(
    SciCore::Real t,
    int r,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePi,
    const std::function<BlockVector<SciCore::Complex>(int, int, SciCore::Real, SciCore::Real)>& computeD_col,
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

    return integrate2DAdaptive(
        [&](Real s, Real q) -> SuperRowVector
        {
            SuperRowVector returnValue = SuperRowVector::Zero(1, dim2);

            Model::BlockDiagonalType Pi_t = computePi(s * (1 - q));

            for (int eta = 0; eta <= 1; ++eta)
            {
                Real etaEff = (eta == 0) ? -1 : 1;
                for (int l1 = 0; l1 < numStates; ++l1)
                {
                    for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
                    {
                        int i    = singleToMultiIndex(static_cast<Eta>(eta), l1, model);
                        int iBar = singleToMultiIndex(static_cast<Eta>(!eta), l1Bar, model);

                        Complex prefactor =
                            -(etaEff / 2) * (s * gammaMinus(s, static_cast<Eta>(eta), l1, l1Bar, r, model));

                        if (prefactor != 0.0)
                        {
                            Model::SuperfermionType D_iBar =
                                effectiveVertexDiagram1_fromCols(iBar, s * q, t - s, computeD_col, block, model);
                            addProduct(
                                prefactor, Tr_superfermionAnnihilation[i], Pi_t, D_iBar, returnValue,
                                blockStartIndices);
                        }
                    }
                }
            }

            return returnValue;
        },
        lower, upper, epsAbs, epsRel);
}

//  _________________________
//  |       _________       |
//  |       |       |       |
//  X-------0-------0-------0
//
Model::SuperRowVectorType currentDiagram_2_2(
    SciCore::Real t,
    int r,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computePi,
    const std::function<Model::BlockDiagonalType(SciCore::Real)>& computeDiagram_1,
    const std::vector<Model::SuperfermionType>& superfermion,
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

    // It appears that this safety factor is sometimes necessary.
    Real safety = 0.1;
    return integrate2DAdaptive(
        [&](Real p, Real q) -> SuperRowVector
        {
            Real t1 = p * t;
            Real t2 = q * t1;

            SuperRowVector returnValue = SuperRowVector::Zero(1, dim2);

            Model::BlockDiagonalType Pi_1      = computePi(t - t1);
            Model::BlockDiagonalType diagram_1 = computeDiagram_1(t1 - t2);
            Model::BlockDiagonalType Pi_2      = computePi(t2);

            Model::BlockDiagonalType middle = product(1.0, Pi_1, diagram_1, Pi_2);

            for (int eta = 0; eta <= 1; ++eta)
            {
                Real etaEff = (eta == 0) ? -1 : 1;
                for (int l1 = 0; l1 < numStates; ++l1)
                {
                    for (int l1Bar = 0; l1Bar < numStates; ++l1Bar)
                    {
                        int i    = singleToMultiIndex(static_cast<Eta>(eta), l1, model);
                        int iBar = singleToMultiIndex(static_cast<Eta>(!eta), l1Bar, model);

                        Complex prefactor =
                            -(etaEff / 2) * t * (t1 * gammaMinus(t, static_cast<Eta>(eta), l1, l1Bar, r, model));

                        if (prefactor != 0.0)
                        {
                            addProduct(
                                prefactor, Tr_superfermionAnnihilation[i], middle, superfermion[iBar], returnValue,
                                blockStartIndices);
                        }
                    }
                }
            }

            return returnValue;
        },
        lower, upper, safety* epsAbs, safety* epsRel);
}

} // namespace RealTimeTransport::RenormalizedPT::Detail
