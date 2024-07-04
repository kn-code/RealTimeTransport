//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/RenormalizedPT/ConductanceKernel.h"

#include "RealTimeTransport/BlockMatrices/BlockDiagonalCheb.h"
#include "RealTimeTransport/BlockMatrices/MatrixExp.h"
#include "RealTimeTransport/RenormalizedPT/ConductanceDiagrams.h"
#include "RealTimeTransport/RenormalizedPT/Diagrams.h"
#include "RealTimeTransport/Utility.h"

#include <Eigen/QR>

namespace RealTimeTransport::RenormalizedPT
{

int ConductanceKernel::r() const noexcept
{
    return _r;
}

SciCore::Real ConductanceKernel::tMax() const
{
    return _d_dmu_currentKernel.upperLimit();
}

SciCore::Real ConductanceKernel::errorGoal() const noexcept
{
    return _errorGoal;
}

SciCore::Real ConductanceKernel::conductance() const
{
    using namespace SciCore;

    Complex returnValue =
        Complex(_currentKernelZeroFreq * _d_dmu_rhoStat) + Complex(_d_dmu_currentKernel.integrate()(tMax()) * _rhoStat);

    return returnValue.real();
}

Model::OperatorType ConductanceKernel::dState() const
{
    return _model->operatorize(_d_dmu_rhoStat);
}

void ConductanceKernel::_initialize(
    const MemoryKernel& K,
    const CurrentKernel& KI,
    const Model::OperatorType& stationaryState,
    int r,
    Order order,
    SciCore::Real tMax,
    SciCore::Real errorGoal,
    tf::Executor* executor,
    int block,
    SciCore::Real hMin)
{
    using namespace SciCore;
    using Operator       = Model::OperatorType;
    using Supervector    = Model::SupervectorType;
    using SuperRowVector = Model::SuperRowVectorType;

    const Model* model            = K.model();
    _model                        = model->copy();
    _r                            = r;
    _errorGoal                    = errorGoal;
    _rhoStat                      = model->vectorize(stationaryState);
    _currentKernelZeroFreq        = KI.zeroFrequency();
    int rI                        = KI.r();
    auto superfermion             = computeAllSuperfermions(Keldysh::Plus, model);
    auto superfermionAnnihilation = computeAllSuperfermions(Keldysh::Minus, model);

    Real epsAbs = _errorGoal;
    Real epsRel = 0;
    Real tCrit  = std::min(2 / maxNorm(K.LInfty()), tMax);
    Real safety = 0.1;

    // Block structure information
    const std::vector<int>& blockDims = model->blockDimensions();
    int numBlocks                     = blockDims.size();
    std::vector<int> blockStartIndices(numBlocks, 0);
    for (int i = 1; i < numBlocks; ++i)
    {
        blockStartIndices[i] = blockStartIndices[i - 1] + blockDims[i - 1];
    }

    if (hMin < 0)
    {
        hMin = defaultMinChebDistance(0, errorGoal);
    }

    // Compute the trace functional as rowvector (idRow)
    int dim              = model->dimHilbertSpace();
    Operator id          = Operator::Identity(dim, dim);
    Supervector idCol    = model->vectorize(id);
    SuperRowVector idRow = idCol.transpose();

    // Compte the functional Tr D^- as rowvector, where D^- is an annihilation superfermion
    std::vector<SuperRowVector> Tr_superfermionAnnihilation;
    Tr_superfermionAnnihilation.reserve(superfermionAnnihilation.size());
    for (const auto& X : superfermionAnnihilation)
    {
        Tr_superfermionAnnihilation.push_back(product(1.0, idRow, X, blockStartIndices));
    }

    BlockDiagonalMatrixExp piInfty(K.LInfty());
    std::function<BlockDiagonalMatrix(Real)> computePiInfty = [&](Real t) -> BlockDiagonalMatrix
    {
        return piInfty(t);
    };

    std::function<BlockDiagonalMatrix(Real)> computePiInftyM1 = [&](Real t) -> BlockDiagonalMatrix
    {
        return piInfty.expm1(t);
    };

    if (order == Order::_1)
    {
        //
        // Compute dΣ/dμ
        //
        std::function<Matrix(int, Real)> d_dmu_memoryKernel_1;
        if (block < 0)
        {
            d_dmu_memoryKernel_1 = [&](int blockIndex, Real t) -> Matrix
            {
                return Detail::d_dmu_diagram_1(blockIndex, t, r, computePiInfty, superfermion, model);
            };
        }
        else
        {
            d_dmu_memoryKernel_1 = [&](int blockIndex, Real t) -> Matrix
            {
                if (blockIndex == block)
                    return Detail::d_dmu_diagram_1(blockIndex, t, r, computePiInfty, superfermion, model);
                else
                    return Matrix::Zero(model->blockDimensions()[blockIndex], model->blockDimensions()[blockIndex]);
            };
        }

        bool ok = false;
        if (executor == nullptr)
        {
            _d_dmu_memoryKernel =
                BlockDiagonalCheb(d_dmu_memoryKernel_1, numBlocks, K.K().sections(), epsAbs, epsRel, hMin, &ok);
        }
        else
        {
            _d_dmu_memoryKernel = BlockDiagonalCheb(
                d_dmu_memoryKernel_1, numBlocks, K.K().sections(), epsAbs, epsRel, hMin, *executor, &ok);
        }

        if (ok == false)
        {
            throw Error("Accuracy goal not reached");
        }

        //
        // Compute dΣ_I/dμ
        //
        auto d_dmu_currentKernel_1 = [&](Real t) -> SuperRowVector
        {
            return Detail::d_dmu_currentDiagram_1(
                t, rI, r, computePiInfty, Tr_superfermionAnnihilation, superfermion, blockStartIndices, model);
        };

        ok = false;
        if (executor == nullptr)
        {
            _d_dmu_currentKernel =
                ChebAdaptive<SuperRowVector>(d_dmu_currentKernel_1, KI.K().sections(), epsAbs, epsRel, hMin, &ok);
        }
        else
        {
            _d_dmu_currentKernel = ChebAdaptive<SuperRowVector>(
                d_dmu_currentKernel_1, KI.K().sections(), epsAbs, epsRel, hMin, *executor, &ok);
        }

        if (ok == false)
        {
            throw Error("Accuracy goal not reached");
        }
    }
    else if (order == Order::_2)
    {
        BlockDiagonalCheb chebDiagram1;
        bool ok = false;
        if (executor == nullptr)
        {
            chebDiagram1 = BlockDiagonalCheb(
                [&](int blockIndex, Real t) -> Matrix {
                    return Detail::diagram_1(
                        blockIndex, t, tCrit, computePiInfty, computePiInftyM1, superfermion, model);
                },
                numBlocks, K.K().sections(), safety* epsAbs, epsRel, hMin, &ok);
        }
        else
        {
            chebDiagram1 = BlockDiagonalCheb(
                [&](int blockIndex, Real t) -> Matrix {
                    return Detail::diagram_1(
                        blockIndex, t, tCrit, computePiInfty, computePiInftyM1, superfermion, model);
                },
                numBlocks, K.K().sections(), safety* epsAbs, epsRel, hMin, *executor, &ok);
        }

        if (ok == false)
        {
            throw Error("Accuracy goal not reached");
        }

        BlockDiagonalCheb cheb_d_dmu_diagram1;
        if (executor == nullptr)
        {
            cheb_d_dmu_diagram1 = BlockDiagonalCheb(
                [&](int blockIndex, Real t) -> Matrix
                { return Detail::d_dmu_diagram_1(blockIndex, t, r, computePiInfty, superfermion, model); }, numBlocks,
                K.K().sections(), safety* epsAbs, epsRel, hMin, &ok);
        }
        else
        {
            cheb_d_dmu_diagram1 = BlockDiagonalCheb(
                [&](int blockIndex, Real t) -> Matrix
                { return Detail::d_dmu_diagram_1(blockIndex, t, r, computePiInfty, superfermion, model); }, numBlocks,
                K.K().sections(), safety* epsAbs, epsRel, hMin, *executor, &ok);
        }

        if (ok == false)
        {
            throw Error("Accuracy goal not reached");
        }

        std::function<BlockVector(int, int, Real, Real)> computeD = [&](int i, int col, Real t, Real s) -> BlockVector
        {
            return Detail::effectiveVertexDiagram1_col(
                i, col, t, s, tCrit, safety * epsAbs, computePiInfty, computePiInftyM1, superfermion, model);
        };

        std::function<BlockVector(int, int, Real, Real, int)> compute_d_dmu_D = [&](int i, int col, Real t, Real s,
                                                                                    int r) -> BlockVector
        {
            return Detail::d_dmu_effectiveVertexDiagram1_col(
                i, col, t, s, r, safety * epsAbs, computePiInfty, superfermion, model);
        };

        //
        // Compute dΣ/dμ
        //
        std::function<Matrix(int, Real)> d_dmu_memoryKernel = [&](int blockIndex, Real t) -> Matrix
        {
            if (block >= 0 && block != blockIndex)
                return Matrix::Zero(model->blockDimensions()[blockIndex], model->blockDimensions()[blockIndex]);

            return Detail::d_dmu_diagram_1(blockIndex, t, r, computePiInfty, superfermion, model) +
                   Detail::d_dmu_diagram_2(
                       blockIndex, t, r, safety * epsAbs, epsRel, computePiInfty, computeD, compute_d_dmu_D,
                       superfermion, model) +
                   Detail::d_dmu_diagram_2_2(
                       blockIndex, t, r, safety * epsAbs, epsRel, computePiInfty, chebDiagram1, cheb_d_dmu_diagram1,
                       superfermion, model);
        };

        if (executor == nullptr)
        {
            _d_dmu_memoryKernel =
                BlockDiagonalCheb(d_dmu_memoryKernel, numBlocks, K.K().sections(), epsAbs, epsRel, hMin, &ok);
        }
        else
        {
            _d_dmu_memoryKernel = BlockDiagonalCheb(
                d_dmu_memoryKernel, numBlocks, K.K().sections(), epsAbs, epsRel, hMin, *executor, &ok);
        }

        if (ok == false)
        {
            throw Error("Accuracy goal not reached");
        }

        //
        // Compute dΣ_I/dμ
        //
        auto d_dmu_currentKernel = [&](Real t) -> SuperRowVector
        {
            return Detail::d_dmu_currentDiagram_1(
                       t, rI, r, computePiInfty, Tr_superfermionAnnihilation, superfermion, blockStartIndices, model) +
                   Detail::d_dmu_currentDiagram_2(
                       t, rI, r, safety * epsAbs, epsRel, computePiInfty, computeD, compute_d_dmu_D,
                       Tr_superfermionAnnihilation, blockStartIndices, block, model) +
                   Detail::d_dmu_currentDiagram_2_2(
                       t, rI, r, safety * epsAbs, epsRel, computePiInfty, chebDiagram1, cheb_d_dmu_diagram1,
                       superfermion, Tr_superfermionAnnihilation, blockStartIndices, model);
        };

        if (executor == nullptr)
        {
            _d_dmu_currentKernel =
                ChebAdaptive<SuperRowVector>(d_dmu_currentKernel, KI.K().sections(), epsAbs, epsRel, hMin, &ok);
        }
        else
        {
            _d_dmu_currentKernel = ChebAdaptive<SuperRowVector>(
                d_dmu_currentKernel, KI.K().sections(), epsAbs, epsRel, hMin, *executor, &ok);
        }

        if (ok == false)
        {
            throw Error("Accuracy goal not reached");
        }
    }
    else
    {
        throw Error("Not implemented");
    }

    //
    // Compute dρ/dμ by solving dΣ/dμ ρ + Σ dρ/dμ = 0, where ρ denotes the stationary state
    // and Σ the memory kernel at zero-freqeuncy. It must be enforced that Tr dρ/dμ = 0.
    //
    // FIXME this only uses the block structure if block==0, otherwise the full equation is solved
    if (block < 0 || block > 0)
    {
        BlockDiagonalMatrix d_dmu_SigmaZeroFreq = _d_dmu_memoryKernel.integrate()(tMax);
        Supervector b                           = -(d_dmu_SigmaZeroFreq * _rhoStat);
        Matrix A                                = K.zeroFrequency().toDense();
        A.row(A.rows() - 1)                     = idRow;
        b[A.rows() - 1]                         = 0.0;
        _d_dmu_rhoStat                          = A.colPivHouseholderQr().solve(b);
    }
    else if (block == 0)
    {
        if (_rhoStat.tail(_rhoStat.size() - blockDims[block]).isZero() == false)
        {
            throw Error("Stationary state does not have the required structure");
        }

        if (idRow.tail(idRow.size() - blockDims[block]).isZero() == false)
        {
            throw Error("Inconsistent block structure");
        }

        Matrix A                   = K.zeroFrequency()(block);
        Matrix d_dmu_SigmaZeroFreq = _d_dmu_memoryKernel.block(block).integrate()(tMax);
        Supervector b              = -(d_dmu_SigmaZeroFreq * _rhoStat.segment(0, blockDims[block]));

        A.row(A.rows() - 1)                   = idRow.segment(0, blockDims[block]);
        b[A.rows() - 1]                       = 0.0;
        _d_dmu_rhoStat                        = Supervector::Zero(dim * dim);
        _d_dmu_rhoStat.head(blockDims[block]) = A.colPivHouseholderQr().solve(b);
    }
    else
    {
        throw Error("Logic error");
    }

    truncToZero(_d_dmu_rhoStat, std::numeric_limits<Real>::epsilon());
}

} // namespace RealTimeTransport::RenormalizedPT
