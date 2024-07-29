//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/IteratedRG/MemoryKernel.h"

#include <sstream>

#include <SciCore/IDECheb.h>

#include "RealTimeTransport/BlockMatrices/MatrixOperations.h"
#include "RealTimeTransport/ComputePropagator.h"
#include "RealTimeTransport/IteratedRG/Diagrams.h"
#include "RealTimeTransport/RenormalizedPT/Diagrams.h"
#include "RealTimeTransport/RenormalizedPT/MemoryKernel.h"
#include "RealTimeTransport/Utility.h"

namespace RealTimeTransport::IteratedRG
{

SciCore::Real minDistance(const SciCore::RealVector& vec)
{
    using namespace SciCore;

    if (vec.size() < 2)
    {
        throw Error("vec must have at least two elements");
    }

    Real minDistance = std::numeric_limits<Real>::max();

    for (int i = 0; i < vec.size() - 1; i++)
    {
        Real dist = vec[i + 1] - vec[i];
        if (dist < minDistance)
        {
            minDistance = dist;
        }
    }

    return minDistance;
}

SciCore::Real minDistance(const std::vector<SciCore::RealVector>& vectors)
{
    using namespace SciCore;

    if (vectors.empty() == true)
    {
        throw Error("vectors mustn't be empty");
    }

    Real overallMinDistance = std::numeric_limits<Real>::max();

    for (const RealVector& vec : vectors)
    {
        Real currentMinDistance = minDistance(vec);
        if (currentMinDistance < overallMinDistance)
        {
            overallMinDistance = currentMinDistance;
        }
    }

    return overallMinDistance;
}

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

void MemoryKernel::initialize(
    const Model* model,
    Order order,
    SciCore::Real tMax,
    SciCore::Real errorGoal,
    tf::Executor* executor,
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

    Real epsAbs = errorGoal;
    Real epsRel = 0;
    Real safety = 0.1;
    Real tCrit  = std::min(2 / maxNorm(_minusILInfty), tMax);

    BlockDiagonalCheb propagator;
    try
    {
        propagator = _initMemoryKernel(order, tMax, epsAbs, executor, hMin, initialChebSections);
    }
    catch (std::exception& e)
    {
        std::stringstream ss;
        ss << "Failed to initialize RG computation of order " << static_cast<int>(order) << ".\nError message: '"
           << e.what() << "'";
        throw Error(ss.str());
    }
    BlockDiagonalCheb propagatorMinusOne = computePropagatorMinusOne(_minusILInfty, _minusIK, errorGoal, tCrit);

    // For two loop computations we put the minimum allowed interval size to a quarter
    // of the interval size from the perturbation theory. This is to avoid the (rare)
    // situation where an interpolation error goal can't be met because intermediate
    // integration errors accumulate too much.
    if (hMin < 0 && order == Order::_2)
    {
        hMin = std::nextafter(minDistance(_minusIK.sections()) / 4, std::numeric_limits<Real>::max());
    }
    // For three loop computations we instead use half the interval size of the two loop computation.
    else if (hMin < 0 && order == Order::_3)
    {
        hMin = std::nextafter(minDistance(_minusIK.sections()) / 2, std::numeric_limits<Real>::max());
    }

    std::function<BlockDiagonal(Real)> computePi = [&](Real t) -> BlockDiagonal
    {
        return propagator(t);
    };

    std::function<BlockDiagonal(Real)> computePiM1 = [&](Real t) -> BlockDiagonal
    {
#ifdef REAL_TIME_TRANSPORT_DEBUG
        if (t > tCrit)
        {
            throw Error("Invalid time argument");
        }
#endif
        return propagatorMinusOne(t);
    };

    std::function<BlockVector(int, int, Real, Real)> computeD_O3_col = [&](int i, int col, Real t,
                                                                           Real s) -> BlockVector
    {
        return RealTimeTransport::RenormalizedPT::Detail::effectiveVertexDiagram1_col(
            i, col, t, s, tCrit, safety * epsAbs, computePi, computePiM1, superfermion, _model.get());
    };

    std::function<BlockVector(int, int, Real, Real)> computeD_O3_O5_col = [&](int i, int col, Real t,
                                                                              Real s) -> BlockVector
    {
        return IteratedRG::Detail::computeEffectiveVertexCorrections_O3_O5_col(
            i, col, t, s, epsAbs, computePi, computeD_O3_col, superfermion, _model.get());
    };

    auto computeNewKernel_2Loop = [&](int blockIndex, Real t) -> Matrix
    {
        // std::cout << "[DBG] computeNewKernel_2Loop: block=" << blockIndex << ", t=" << t << std::endl;
        return RenormalizedPT::Detail::diagram_1(blockIndex, t, tCrit, computePi, computePiM1, superfermion, model) +
               RenormalizedPT::Detail::diagram_2(
                   blockIndex, t, safety * epsAbs, epsRel, computePi, computeD_O3_col, superfermion, model);
    };

    auto computeNewKernel_3Loop = [&](int blockIndex, Real t) -> Matrix
    {
        // std::cout << "[DBG] computeNewKernel_3Loop: block=" << blockIndex << ", t=" << t << std::endl;
        return RenormalizedPT::Detail::diagram_1(blockIndex, t, tCrit, computePi, computePiM1, superfermion, model) +
               RenormalizedPT::Detail::diagram_2(
                   blockIndex, t, epsAbs, epsRel, computePi, computeD_O3_O5_col, superfermion, model);
    };

    int iteration     = 0;
    int maxIterations = 10;
    Error accuracyError;
    bool hasAccuracyError = false;
    while (true)
    {
        ++iteration;
        if (iteration > maxIterations)
        {
            if (hasAccuracyError == false)
            {
                hasAccuracyError = true;
                accuracyError    = Error("Failed to converge within maximum number of iterations.");
            }

            break;
        }

        BlockDiagonalCheb newMinusIK;

        if (order == Order::_2)
        {
            bool ok = false;
            if (executor == nullptr)
            {
                newMinusIK = BlockDiagonalCheb(
                    computeNewKernel_2Loop, _minusILInfty.numBlocks(), _minusIK.sections(), epsAbs, epsRel, hMin, &ok);
            }
            else
            {
                newMinusIK = BlockDiagonalCheb(
                    computeNewKernel_2Loop, _minusILInfty.numBlocks(), _minusIK.sections(), epsAbs, epsRel, hMin,
                    *executor, &ok);
            }

            if (ok == false && hasAccuracyError == false)
            {
                hasAccuracyError = true;
                std::stringstream ss;
                ss << "Accuracy goal not reached in two loop memory kernel computation.\nIteration: " << iteration
                   << "\nSections:\n";
                for (const auto& sec : newMinusIK.sections())
                {
                    ss << sec.transpose() << "\n";
                }
                accuracyError = Error(ss.str());
            }
        }
        else if (order == Order::_3)
        {
            bool ok = false;
            if (executor == nullptr)
            {
                newMinusIK = BlockDiagonalCheb(
                    computeNewKernel_3Loop, _minusILInfty.numBlocks(), _minusIK.sections(), epsAbs, epsRel, hMin, &ok);
            }
            else
            {
                newMinusIK = BlockDiagonalCheb(
                    computeNewKernel_3Loop, _minusILInfty.numBlocks(), _minusIK.sections(), epsAbs, epsRel, hMin,
                    *executor, &ok);
            }

            if (ok == false && hasAccuracyError == false)
            {
                hasAccuracyError = true;
                std::stringstream ss;
                ss << "Accuracy goal not reached in three loop memory kernel computation.\nIteration: " << iteration
                   << "\nSections:\n";
                for (const auto& sec : newMinusIK.sections())
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

        // Estimate error between _minusIK and newMinusIK
        Real iterationError = 0;
        RealVector tTest    = RealVector::LinSpaced(2 * newMinusIK.block(0).numCoefficients().sum(), 0, tMax);
        for (Real t : tTest)
        {
            BlockDiagonal errMatrix  = newMinusIK(t);
            errMatrix               -= _minusIK(t);
            Real err                 = maxNorm(errMatrix);

            iterationError = std::max(iterationError, err);
        }

        // std::cout << "Iterated with error " << iterationError << std::endl;

        // Update memory kernel
        _minusIK = std::move(newMinusIK);

        if (iterationError < 10 * epsAbs)
        {
            // Iteration was successful --> exit
            break;
        }
        else
        {
            // Iteration was not successful --> update propagator
            Real safetyPi = 0.01;
            int nMinCheb  = 3 * 16;
            int numBlocks = model->blockDimensions().size();
            std::vector<ChebAdaptive<Matrix>> newPropagatorBlocks;
            newPropagatorBlocks.reserve(numBlocks);
            for (int blockIndex = 0; blockIndex < numBlocks; ++blockIndex)
            {
                newPropagatorBlocks.emplace_back(computePropagatorIde(
                    _minusILInfty(blockIndex), _minusIK.block(blockIndex), 0, tMax, safetyPi * epsAbs, epsRel,
                    nMinCheb));
            }

            propagator         = BlockDiagonalCheb(std::move(newPropagatorBlocks));
            propagatorMinusOne = computePropagatorMinusOne(_minusILInfty, _minusIK, errorGoal, tCrit);
        }
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

BlockDiagonalCheb MemoryKernel::_initMemoryKernel(
    Order order,
    SciCore::Real tMax,
    SciCore::Real errorGoal,
    tf::Executor* executor,
    SciCore::Real hMin,
    const std::vector<SciCore::RealVector>* initialChebSections)
{
    using namespace SciCore;

    // Two loop RG is initialized with perturbation theory, but three loop RG is initialized with 2 loop RG
    if (order == Order::_2)
    {
        int block = -1;
        RealTimeTransport::RenormalizedPT::MemoryKernel KPert;
        KPert.initialize(
            _model.get(), RenormalizedPT::Order::_2, tMax, errorGoal, executor, block, hMin, initialChebSections);

        auto Pi                      = computePropagator(KPert);
        BlockDiagonalCheb propagator = std::move(Pi.Pi());
        _minusIK                     = std::move(KPert.K());

        return propagator;
    }
    else if (order == Order::_3)
    {
        MemoryKernel KTwoLoop;
        KTwoLoop.initialize(_model.get(), Order::_2, tMax, errorGoal, executor, hMin, initialChebSections);
        Propagator fullPi            = computePropagator(KTwoLoop);
        BlockDiagonalCheb propagator = std::move(fullPi.Pi());
        _minusIK                     = std::move(KTwoLoop.K());

        return propagator;
    }
    else
    {
        throw Error("Not implemented");
    }
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

} // namespace RealTimeTransport::IteratedRG

namespace RealTimeTransport
{

Propagator computePropagator(const IteratedRG::MemoryKernel& memoryKernel)
{
    return Propagator(memoryKernel.model(), computePropagatorTemplate(memoryKernel, -1));
}

} // namespace RealTimeTransport
