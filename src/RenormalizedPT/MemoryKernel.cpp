//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/RenormalizedPT/MemoryKernel.h"

#include <sstream>

#include <SciCore/IDECheb.h>

#include "RealTimeTransport/BlockMatrices/MatrixExp.h"
#include "RealTimeTransport/BlockMatrices/MatrixOperations.h"
#include "RealTimeTransport/ComputePropagator.h"
#include "RealTimeTransport/RenormalizedPT/Diagrams.h"
#include "RealTimeTransport/Utility.h"

namespace RealTimeTransport::RenormalizedPT
{

MemoryKernel::MemoryKernel() noexcept
{
}

MemoryKernel::MemoryKernel(MemoryKernel&& other) noexcept
    : _model(std::move(other._model)), _errorGoal(other._errorGoal), _minusILInfty(std::move(other._minusILInfty)),
      _minusIK(std::move(other._minusIK))
{
}

MemoryKernel::MemoryKernel(const MemoryKernel& other)
    : _model(nullptr), _errorGoal(other._errorGoal), _minusILInfty(other._minusILInfty), _minusIK(other._minusIK)
{
    if (other._model.get() != nullptr)
    {
        _model = other._model->copy();
    }
}

MemoryKernel& MemoryKernel::operator=(MemoryKernel&& other)
{
    _model        = std::move(other._model);
    _errorGoal    = other._errorGoal;
    _minusILInfty = std::move(other._minusILInfty);
    _minusIK      = std::move(other._minusIK);

    return *this;
}

MemoryKernel& MemoryKernel::operator=(const MemoryKernel& other)
{
    if (other._model.get() != nullptr)
    {
        _model = other._model->copy();
    }
    else
    {
        _model.reset();
    }

    _errorGoal    = other._errorGoal;
    _minusILInfty = other._minusILInfty;
    _minusIK      = other._minusIK;

    return *this;
}

void MemoryKernel::initialize(
    const Model* model,
    Order order,
    SciCore::Real tMax,
    SciCore::Real errorGoal,
    tf::Executor* executor,
    int block,
    SciCore::Real hMin,
    const std::vector<SciCore::RealVector>* initialChebSections)
{
    using namespace SciCore;
    using namespace RealTimeTransport;
    using BlockDiagonal = BlockDiagonalMatrix;

    _model     = model->copy();
    _errorGoal = errorGoal;

    auto superfermion             = computeAllSuperfermions(Keldysh::Plus, model);
    auto superfermionAnnihilation = computeAllSuperfermions(Keldysh::Minus, model);

    _minusILInfty = computeLiouvillian(model) + computeSigmaInfty(superfermion, superfermionAnnihilation, model);
    int numBlocks = _minusILInfty.numBlocks();

    Real epsAbs = errorGoal;
    Real epsRel = 0;
    Real tCrit  = std::min(2 / maxNorm(_minusILInfty), tMax);
    Real safety = 0.1;

    if (hMin < 0)
    {
        hMin = defaultMinChebDistance(tCrit, errorGoal);
    }

    Error accuracyError;
    bool hasAccuracyError = false;

    std::vector<RealVector> initialSections =
        (initialChebSections == nullptr) ? std::vector<RealVector>(numBlocks, defaultInitialChebSections(tMax, tCrit))
                                         : *initialChebSections;

    BlockDiagonalMatrixExp piInfty(_minusILInfty);

    std::function<BlockDiagonal(Real)> computePiInfty = [&](Real t) -> BlockDiagonal
    {
        return piInfty(t);
    };

    std::function<BlockDiagonal(Real)> computePiInftyM1 = [&](Real t) -> BlockDiagonal
    {
        return piInfty.expm1(t);
    };

    if (order == Order::_1)
    {
        auto diagram_1_full = [&](int blockIndex, Real t) -> Matrix
        {
            return Detail::diagram_1(blockIndex, t, tCrit, computePiInfty, computePiInftyM1, superfermion, model);
        };

        auto diagram_1_block = [&](int blockIndex, Real t) -> Matrix
        {
            if (blockIndex == block)
                return Detail::diagram_1(blockIndex, t, tCrit, computePiInfty, computePiInftyM1, superfermion, model);
            else
                return Matrix::Zero(model->blockDimensions()[blockIndex], model->blockDimensions()[blockIndex]);
        };

        std::function<Matrix(int, Real)> diagram_1;
        if (block < 0)
        {
            diagram_1 = diagram_1_full;
        }
        else
        {
            diagram_1 = diagram_1_block;
        }

        bool ok = false;
        if (executor == nullptr)
        {
            _minusIK = BlockDiagonalCheb(diagram_1, numBlocks, initialSections, errorGoal, 0.0, hMin, &ok);
        }
        else
        {
            _minusIK = BlockDiagonalCheb(diagram_1, numBlocks, initialSections, errorGoal, 0.0, hMin, *executor, &ok);
        }

        if (ok == false && hasAccuracyError == false)
        {
            hasAccuracyError = true;
            std::stringstream ss;
            ss << "First order renormalized memory kernel computation failed. Sections:\n";
            for (const auto& sec : _minusIK.sections())
            {
                ss << sec.transpose() << "\n";
            }
            accuracyError = Error(ss.str());
        }
    }
    else if (order == Order::_2)
    {
        bool ok = false;

        BlockDiagonalCheb chebFullDiagram1;
        if (executor == nullptr)
        {
            chebFullDiagram1 = BlockDiagonalCheb(
                [&](int blockIndex, Real t) -> Matrix {
                    return Detail::diagram_1(
                        blockIndex, t, tCrit, computePiInfty, computePiInftyM1, superfermion, model);
                },
                numBlocks, initialSections, safety* epsAbs, epsRel, hMin, &ok);
        }
        else
        {
            chebFullDiagram1 = BlockDiagonalCheb(
                [&](int blockIndex, Real t) -> Matrix {
                    return Detail::diagram_1(
                        blockIndex, t, tCrit, computePiInfty, computePiInftyM1, superfermion, model);
                },
                numBlocks, initialSections, safety* epsAbs, epsRel, hMin, *executor, &ok);
        }

        if (ok == false && hasAccuracyError == false)
        {
            hasAccuracyError = true;
            std::stringstream ss;
            ss << "Second order renormalized memory kernel computation failed [chebFullDiagram1]. Sections:\n";
            for (const auto& sec : chebFullDiagram1.sections())
            {
                ss << sec.transpose() << "\n";
            }
            accuracyError = Error(ss.str());
        }

        std::function<BlockVector(int, int, Real, Real)> computeD = [&](int i, int col, Real t, Real s) -> BlockVector
        {
            return Detail::effectiveVertexDiagram1_col(
                i, col, t, s, tCrit, safety * epsAbs, computePiInfty, computePiInftyM1, superfermion, model);
        };

        auto diagram_2_1 = [&](int blockIndex, Real t) -> Matrix
        {
            return Detail::diagram_2(
                blockIndex, t, safety * epsAbs, epsRel, computePiInfty, computeD, superfermion, model);
        };

        auto diagram_2_2 = [&](int blockIndex, Real t) -> Matrix
        {
            return Detail::diagram_2_2(
                blockIndex, t, safety * epsAbs, epsRel, computePiInfty, chebFullDiagram1, superfermion, model);
        };

        // Set up initial second order sections
        if (block < 0)
        {
            initialSections = chebFullDiagram1.sections();
        }
        else
        {
            for (size_t i = 0; i < initialSections.size(); ++i)
            {
                if (static_cast<int>(i) == block)
                {
                    initialSections[i] = chebFullDiagram1.sections()[i];
                }
                else
                {
                    initialSections[i] = RealVector{
                        {0, tMax}
                    };
                }
            }
        }

        // Even though it can be faster to compute diagram_2_1/diagram_2_2 separately into their own Cheb objects
        // first and then assemble the result at the end, we don't do this because it can lose accuracy such that
        // the computation fails (depending on parameters).
        if (executor == nullptr)
        {
            _minusIK = BlockDiagonalCheb(
                [&](int blockIndex, Real t) -> Matrix
                {
                    if (block < 0 || blockIndex == block)
                    {
                        return chebFullDiagram1(blockIndex, t) + diagram_2_1(blockIndex, t) +
                               diagram_2_2(blockIndex, t);
                    }
                    else
                    {
                        return Matrix::Zero(model->blockDimensions()[blockIndex], model->blockDimensions()[blockIndex]);
                    }
                },
                numBlocks, initialSections, epsAbs, epsRel, hMin, &ok);
        }
        else
        {
            _minusIK = BlockDiagonalCheb(
                [&](int blockIndex, Real t) -> Matrix
                {
                    if (block < 0 || blockIndex == block)
                    {
                        return chebFullDiagram1(blockIndex, t) + diagram_2_1(blockIndex, t) +
                               diagram_2_2(blockIndex, t);
                    }
                    else
                    {
                        return Matrix::Zero(model->blockDimensions()[blockIndex], model->blockDimensions()[blockIndex]);
                    }
                },
                numBlocks, initialSections, epsAbs, epsRel, hMin, *executor, &ok);
        }

        if (ok == false && hasAccuracyError == false)
        {
            hasAccuracyError = true;
            std::stringstream ss;
            ss << "Accuracy goal not reached in renormalized memory kernel computation. Sections:\n";
            for (const auto& sec : _minusIK.sections())
            {
                ss << sec.transpose() << "\n";
            }
            accuracyError = Error(ss.str());
        }
    }
    else
    {
        throw Error("Not implemented");
    }

    if (hasAccuracyError == true)
    {
        MemoryKernel preliminaryResult;
        preliminaryResult._model        = std::move(_model);
        preliminaryResult._errorGoal    = _errorGoal;
        preliminaryResult._minusILInfty = std::move(_minusILInfty);
        preliminaryResult._minusIK      = std::move(_minusIK);

        throw AccuracyError<MemoryKernel>(std::move(accuracyError), std::move(preliminaryResult));
    }
}

bool MemoryKernel::operator==(const MemoryKernel& other) const noexcept
{
    bool modelSame = false;
    if (_model.get() == nullptr && other._model.get() == nullptr)
    {
        modelSame = true;
    }
    else if (_model.get() != nullptr && other._model.get() != nullptr)
    {
        modelSame = ((*_model) == (*other._model));
    }

    return modelSame && (_errorGoal == other._errorGoal) && (_minusILInfty == other._minusILInfty) &&
           (_minusIK == other._minusIK);
}

bool MemoryKernel::operator!=(const MemoryKernel& other) const noexcept
{
    return !operator==(other);
}

const Model* MemoryKernel::model() const noexcept
{
    return _model.get();
}

SciCore::Real MemoryKernel::tMax() const
{
    return _minusIK.upperLimit();
}

SciCore::Real MemoryKernel::errorGoal() const noexcept
{
    return _errorGoal;
}

const BlockDiagonalMatrix& MemoryKernel::LInfty() const noexcept
{
    return _minusILInfty;
}

BlockDiagonalCheb& MemoryKernel::K() noexcept
{
    return _minusIK;
}

const BlockDiagonalCheb& MemoryKernel::K() const noexcept
{
    return _minusIK;
}

BlockDiagonalMatrix MemoryKernel::zeroFrequency() const
{
    BlockDiagonalMatrix returnValue  = _minusILInfty;
    returnValue                     += _minusIK.integrate()(tMax());
    return returnValue;
}

Model::OperatorType MemoryKernel::stationaryState(int block) const
{
    using namespace SciCore;

    Real tol = 1000 * std::numeric_limits<Real>::epsilon();
    if (block < 0)
    {
        return computeStationaryState(zeroFrequency(), _model, tol);
    }
    else
    {
        return computeStationaryState(
            _minusILInfty(block) + _minusIK.block(block).integrate()(tMax()), _model, block, tol);
    }
}

} // namespace RealTimeTransport::RenormalizedPT

namespace RealTimeTransport
{

Propagator computePropagator(const RenormalizedPT::MemoryKernel& memoryKernel, int block)
{
    return Propagator(memoryKernel.model(), computePropagatorTemplate(memoryKernel, block));
}

} // namespace RealTimeTransport
