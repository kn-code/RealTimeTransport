//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/IteratedRG/ConductanceDiagrams.h"

#include <SciCore/Integration.h>

#include "RealTimeTransport/BlockMatrices/MatrixOperations.h"
#include "RealTimeTransport/Utility.h"

namespace RealTimeTransport::IteratedRG::Detail
{

//  _________________________
//  |       ____/____       |
//  |       |       |       |
//  0-------0-------0-------0
//
SciCore::Matrix d_dmu_diagram_2_2(
    int blockIndex,
    SciCore::Real t,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
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
            BlockDiagonalMatrix d_dmu_diagram_1 = compute_d_dmu_diagram_1(t1 - t2);
            BlockDiagonalMatrix Pi_2            = computePi(t2);

            BlockDiagonalMatrix middle = product(1.0, Pi_1, d_dmu_diagram_1, Pi_2);

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
        lower, upper, epsAbs, epsRel);
}

//  _________________________
//  |       ____/____       |
//  |       |       |       |
//  X-------0-------0-------0
//
Model::SuperRowVectorType d_dmu_currentDiagram_2_2(
    SciCore::Real t,
    int rI,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
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
            BlockDiagonalMatrix d_dmu_diagram_1 = compute_d_dmu_diagram_1(t1 - t2);
            BlockDiagonalMatrix Pi_2            = computePi(t2);

            BlockDiagonalMatrix middle = product(1.0, Pi_1, d_dmu_diagram_1, Pi_2);

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
                            -(etaEff / 2) * t * (t1 * gammaMinus(t, static_cast<Eta>(eta), l1, l1Bar, rI, model));

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
        lower, upper, epsAbs, epsRel);
}

} // namespace RealTimeTransport::IteratedRG::Detail
