//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include <iostream> // FIXME delete
#include <sstream>

#include <SciCore/IDECheb.h>

#include "RealTimeTransport/BlockMatrices/MatrixOperations.h"
#include "RealTimeTransport/ComputePropagator.h"
#include "RealTimeTransport/IteratedRG/Diagrams.h"
#include "RealTimeTransport/IteratedRG/MemoryKernel.h"
#include "RealTimeTransport/RenormalizedPT/Diagrams.h"
#include "RealTimeTransport/RenormalizedPT/MemoryKernel.h"
#include "RealTimeTransport/Utility.h"

namespace RealTimeTransport::IteratedRG
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

const Model::BlockDiagonalType& MemoryKernel::LInfty() const noexcept
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
    using BlockDiagonal = Model::BlockDiagonalType;

    _model     = model->copy();
    _errorGoal = errorGoal;

    auto superfermion             = computeAllSuperfermions(Keldysh::Plus, model);
    auto superfermionAnnihilation = computeAllSuperfermions(Keldysh::Minus, model);

    _minusILInfty = computeLiouvillian(model) + computeSigmaInfty(superfermion, superfermionAnnihilation, model);

    Real epsAbs = errorGoal;
    Real epsRel = 0;
    Real tCrit  = std::min(2 / maxNorm(_minusILInfty), tMax);
    Real safety = 0.1;

    if (hMin < 0)
    {
        hMin = defaultMinChebDistance(tCrit, errorGoal);
    }

    BlockDiagonalCheb propagator         = _initMemoryKernel(order, tMax, epsAbs, executor, hMin, initialChebSections);
    BlockDiagonalCheb propagatorMinusOne = computePropagatorMinusOne(_minusILInfty, _minusIK, errorGoal, tCrit);

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

    std::function<BlockVector<Complex>(int, int, Real, Real)> computeD_O3_col = [&](int i, int col, Real t,
                                                                                    Real s) -> BlockVector<Complex>
    {
        return RealTimeTransport::RenormalizedPT::Detail::effectiveVertexDiagram1_col(
            i, col, t, s, safety * epsAbs, tCrit, computePi, computePiM1, superfermion, _model.get());
    };

    std::function<BlockVector<Complex>(int, int, Real, Real)> computeD_O3_O5_col = [&](int i, int col, Real t,
                                                                                       Real s) -> BlockVector<Complex>
    {
        // FIXME is a safety factor also needed here ?
        return _computeD_O3_O5_col(i, col, t, s, epsAbs, computePi, computeD_O3_col, superfermion);
    };

    auto computeNewKernel_2Loop = [&](int blockIndex, Real t) -> Matrix
    {
        std::cout << "computeNewKernel_2Loop: " << blockIndex << ", " << t << std::endl;
        return RenormalizedPT::Detail::diagram_1(blockIndex, t, tCrit, computePi, computePiM1, superfermion, model) +
               RenormalizedPT::Detail::diagram_2(
                   blockIndex, t, safety * epsAbs, epsRel, computePi, computeD_O3_col, superfermion, model);
    };

    auto computeNewKernel_3Loop = [&](int blockIndex, Real t) -> Matrix
    {
        // FIXME is a safety factor also needed here ?
        std::cout << "computeNewKernel_3Loop: " << blockIndex << ", " << t << std::endl;
        return RenormalizedPT::Detail::diagram_1(blockIndex, t, tCrit, computePi, computePiM1, superfermion, model) +
               RenormalizedPT::Detail::diagram_2(
                   blockIndex, t, epsAbs, epsRel, computePi, computeD_O3_O5_col, superfermion, model);
    };

    int iteration     = 0;
    int maxIterations = 10;
    while (true)
    {
        ++iteration;
        if (iteration > maxIterations)
        {
            throw Error("Failed to converge within maximum number of iterations.");
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

            if (ok == false)
            {
                std::stringstream ss;
                ss << "Two loop memory kernel computation failed. Sections:\n";
                for (const auto& sec : newMinusIK.sections())
                {
                    ss << sec.transpose() << "\n";
                }
                throw Error(ss.str());
            }
        }
        else if (order == Order::_3)
        {
            bool ok = false;
            std::cout << "Start iteration with sections " << _minusIK.sections()[0].transpose() << "\n";
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

            if (ok == false)
            {
                std::stringstream ss;
                ss << "Three loop memory kernel computation failed. Sections:\n";
                for (const auto& sec : newMinusIK.sections())
                {
                    ss << sec.transpose() << "\n";
                }
                throw Error(ss.str());
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

        std::cout << "Iterated with error " << iterationError << std::endl;

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
            int nMinCheb  = 3 * 16;
            int numBlocks = model->blockDimensions().size();
            std::vector<ChebAdaptive<Matrix>> newPropagatorBlocks;
            newPropagatorBlocks.reserve(numBlocks);
            for (int blockIndex = 0; blockIndex < numBlocks; ++blockIndex)
            {
                newPropagatorBlocks.emplace_back(computePropagatorIde(
                    _minusILInfty(blockIndex), _minusIK.block(blockIndex), 0, tMax, safety * epsAbs, epsRel, nMinCheb));
            }

            propagator         = BlockDiagonalCheb(std::move(newPropagatorBlocks));
            propagatorMinusOne = computePropagatorMinusOne(_minusILInfty, _minusIK, errorGoal, tCrit);
        }
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
        // This safety factor initializes the perturbation theory more accurately then the requested error goal.
        // This seems to be needed, otherwise the iteration later on can fail.
        int block   = -1;
        Real safety = 0.1;
        RealTimeTransport::RenormalizedPT::MemoryKernel KPert;
        KPert.initialize(
            _model.get(), RenormalizedPT::Order::_2, tMax, safety * errorGoal, executor, block, hMin,
            initialChebSections);

        auto Pi                      = computePropagator(KPert);
        BlockDiagonalCheb propagator = std::move(Pi.Pi());
        _minusIK                     = std::move(KPert.K());

        std::cout << "Initialized with perturbation theory, " << "mu = " << _model->chemicalPotentials().transpose()
                  << "\n";
        std::cout << "Sections =\n";
        for (const auto& sec : _minusIK.sections())
        {
            std::cout << sec.transpose() << "\n";
        }
        std::cout << std::endl;
        return propagator;
    }
    else if (order == Order::_3)
    {
        // FIXME is a safety factor also needed here ?
        MemoryKernel KTwoLoop;
        KTwoLoop.initialize(_model.get(), Order::_2, tMax, errorGoal, executor, hMin, initialChebSections);
        Propagator fullPi            = computePropagator(KTwoLoop);
        BlockDiagonalCheb propagator = std::move(fullPi.Pi());
        _minusIK                     = std::move(KTwoLoop.K());

        std::cout << "Initialized with 2 loop RG" << std::endl;
        return propagator;
    }
    else
    {
        throw Error("Not implemented");
    }
}

BlockVector<SciCore::Complex> MemoryKernel::_computeD_O3_O5_col(
    int i,
    int col,
    SciCore::Real t_minus_tau,
    SciCore::Real tau_minus_s,
    SciCore::Real epsAbs,
    const std::function<BlockDiagonalMatrix(SciCore::Real)>& computePi,
    const std::function<BlockVector<SciCore::Complex>(int, int, SciCore::Real, SciCore::Real)>& computeD_O3_col,
    const std::vector<Model::SuperfermionType>& superfermion)
{
    using namespace SciCore;

    BlockVector<Complex> returnValue = computeD_O3_col(i, col, t_minus_tau, tau_minus_s);

    returnValue += Detail::effectiveVertexCorrection1_col(
        i, col, t_minus_tau, tau_minus_s, epsAbs, computePi, computeD_O3_col, superfermion, _model.get());

    returnValue += Detail::effectiveVertexCorrection2_col(
        i, col, t_minus_tau, tau_minus_s, epsAbs, computePi, computeD_O3_col, superfermion, _model.get());

    returnValue += Detail::effectiveVertexCorrection3_col(
        i, col, t_minus_tau, tau_minus_s, epsAbs, computePi, superfermion, _model.get());

    returnValue += Detail::effectiveVertexCorrection4_col(
        i, col, t_minus_tau, tau_minus_s, epsAbs, computePi, superfermion, _model.get());

    return returnValue;
}

Model::BlockDiagonalType MemoryKernel::zeroFrequency() const
{
    Model::BlockDiagonalType returnValue  = _minusILInfty;
    returnValue                          += _minusIK.integrate()(tMax());
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
