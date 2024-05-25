//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include <sstream>

#include <SciCore/Utility.h>

#include "RealTimeTransport/BlockMatrices/MatrixOperations.h"
#include "RealTimeTransport/ComputePropagator.h"
#include "RealTimeTransport/IteratedRG/CurrentKernel.h"
#include "RealTimeTransport/IteratedRG/Diagrams.h"
#include "RealTimeTransport/RenormalizedPT/Diagrams.h"
#include "RealTimeTransport/Utility.h"

namespace RealTimeTransport::IteratedRG
{

const Model* CurrentKernel::model() const noexcept
{
    return _model.get();
}

SciCore::Real CurrentKernel::tMax() const
{
    return _minusIK.upperLimit();
}

SciCore::Real CurrentKernel::errorGoal() const noexcept
{
    return _errorGoal;
}

const Model::SuperRowVectorType& CurrentKernel::SigmaInfty() const noexcept
{
    return _minusISigmaInfty;
}

const SciCore::ChebAdaptive<Model::SuperRowVectorType>& CurrentKernel::K() const noexcept
{
    return _minusIK;
}

Model::SuperRowVectorType CurrentKernel::zeroFrequency() const
{
    return _minusISigmaInfty + _minusIK.integrate()(tMax());
}

SciCore::Real CurrentKernel::stationaryCurrent(const Model::OperatorType& stationaryState) const
{
    const auto& rho              = _model->vectorize(stationaryState);
    SciCore::Complex returnValue = zeroFrequency() * rho;
    return returnValue.real();
}

void CurrentKernel::_initialize(
    const MemoryKernel& K,
    const Propagator& propagator,
    int r,
    Order order,
    SciCore::Real tMax,
    SciCore::Real errorGoal,
    tf::Executor* executor,
    int block)
{
    using namespace SciCore;
    using namespace RealTimeTransport;
    using Operator       = Model::OperatorType;
    using Supervector    = Model::SupervectorType;
    using SuperRowVector = Model::SuperRowVectorType;

    _model     = K.model()->copy();
    _errorGoal = errorGoal;

    // Compute superfermions
    auto superfermion             = computeAllSuperfermions(Keldysh::Plus, _model);
    auto superfermionAnnihilation = computeAllSuperfermions(Keldysh::Minus, _model);

    // Compute infinite temperature current kernel
    _minusISigmaInfty = computeSigmaInftyCurrent(r, superfermionAnnihilation, _model);
    Real tCrit        = std::min(2 / maxNorm(K.LInfty()), tMax);

    std::function<BlockDiagonalMatrix(Real)> computePi = [&](Real t) -> BlockDiagonalMatrix
    {
        return propagator.Pi()(t);
    };

    BlockDiagonalCheb propagatorMinusOne = computePropagatorMinusOne(K.LInfty(), K.K(), errorGoal, tCrit);
    std::function<BlockDiagonalMatrix(Real)> computePiM1 = [&](Real t) -> BlockDiagonalMatrix
    {
#ifdef REAL_TIME_TRANSPORT_DEBUG
        if (t > tCrit)
        {
            throw Error("Invalid time argument");
        }
#endif
        return propagatorMinusOne(t);
    };

    // Compute the trace functional as rowvector (idRow)
    int dim              = _model->dimHilbertSpace();
    Operator id          = Operator::Identity(dim, dim);
    Supervector idCol    = _model->vectorize(id);
    SuperRowVector idRow = idCol.transpose();

    // Set up information about block structure
    const std::vector<int>& blockDims = _model->blockDimensions();
    std::vector<int> blockStartIndices(blockDims.size(), 0);
    for (size_t i = 1; i < blockDims.size(); ++i)
    {
        blockStartIndices[i] = blockStartIndices[i - 1] + blockDims[i - 1];
    }

    // Compte the functional Tr D^- as rowvector, where D^- is an annihilation superfermion
    std::vector<SuperRowVector> Tr_superfermionAnnihilation;
    Tr_superfermionAnnihilation.reserve(superfermionAnnihilation.size());
    for (const auto& X : superfermionAnnihilation)
    {
        Tr_superfermionAnnihilation.push_back(product(1.0, idRow, X, blockStartIndices));
    }

    RealVector sections = defaultInitialChebSections(tMax, tCrit);

    Real epsAbs = errorGoal;
    Real epsRel = 0;
    Real safety = 0.1;
    Real hMin   = defaultMinChebDistance(tCrit, errorGoal);

    auto diagram_1 = [&](Real t) -> SuperRowVector
    {
        return RenormalizedPT::Detail::currentDiagram_1(
            t, r, tCrit, computePi, computePiM1, idRow, Tr_superfermionAnnihilation, superfermion, blockStartIndices,
            _model.get());
    };

    bool ok = false;
    ChebAdaptive<SuperRowVector> chebDiagram1;
    if (executor == nullptr)
    {
        chebDiagram1 = ChebAdaptive<SuperRowVector>(diagram_1, sections, safety * epsAbs, epsRel, hMin, &ok);
    }
    else
    {
        chebDiagram1 = ChebAdaptive<SuperRowVector>(diagram_1, sections, safety * epsAbs, epsRel, hMin, *executor, &ok);
    }

    if (ok == false)
    {
        std::stringstream ss;
        ss << "Current kernel computation failed [chebDiagram1]. Sections: " << chebDiagram1.sections().transpose();
        throw Error(ss.str());
    }

    std::function<BlockVector(int, int, Real, Real)> computeD_O3_col = [&](int i, int col, Real t,
                                                                           Real s) -> BlockVector
    {
        return RenormalizedPT::Detail::effectiveVertexDiagram1_col(
            i, col, t, s, tCrit, safety * epsAbs, computePi, computePiM1, superfermion, _model.get());
    };

    std::function<BlockVector(int, int, Real, Real)> computeD_col;
    if (order == Order::_2)
    {
        computeD_col = computeD_O3_col;
    }
    else if (order == Order::_3)
    {
        computeD_col = [&](int i, int col, Real t, Real s) -> BlockVector
        {
            BlockVector returnValue = computeD_O3_col(i, col, t, s);

            returnValue += Detail::effectiveVertexCorrection1_col(
                i, col, t, s, epsAbs, computePi, computeD_O3_col, superfermion, _model.get());

            returnValue += Detail::effectiveVertexCorrection2_col(
                i, col, t, s, epsAbs, computePi, computeD_O3_col, superfermion, _model.get());

            returnValue +=
                Detail::effectiveVertexCorrection3_col(i, col, t, s, epsAbs, computePi, superfermion, _model.get());

            returnValue +=
                Detail::effectiveVertexCorrection4_col(i, col, t, s, epsAbs, computePi, superfermion, _model.get());

            return returnValue;
        };
    }
    else
    {
        throw Error("Not implemented");
    }

    auto diagram_2 = [&](Real t) -> SuperRowVector
    {
        std::cout << "Eval current diagram_2 at t=" << t << std::endl;

        // FIXME test: do we need a safety factor here ?
        return RenormalizedPT::Detail::currentDiagram_2(
            t, r, epsAbs, epsRel, computePi, computeD_col, Tr_superfermionAnnihilation, blockStartIndices, block,
            _model.get());
    };

    if (executor == nullptr)
    {
        _minusIK = ChebAdaptive<SuperRowVector>(
            [&](Real t) -> SuperRowVector { return chebDiagram1(t) + diagram_2(t); }, chebDiagram1.sections(), epsAbs,
            epsRel, hMin, &ok);
    }
    else
    {
        _minusIK = ChebAdaptive<SuperRowVector>(
            [&](Real t) -> SuperRowVector { return chebDiagram1(t) + diagram_2(t); }, chebDiagram1.sections(), epsAbs,
            epsRel, hMin, *executor, &ok);
    }

    if (ok == false)
    {
        std::stringstream ss;
        ss << "Current kernel computation failed. Sections: " << _minusIK.sections().transpose();
        throw Error(ss.str());
    }
}

} // namespace RealTimeTransport::IteratedRG

namespace RealTimeTransport
{

// FIXME this is exactly the same code as in renormalized PT, make it a template
SciCore::ChebAdaptive<SciCore::Real> computeCurrent(
    const IteratedRG::CurrentKernel& KCurrent,
    const Propagator& propagator,
    const Model::OperatorType& rho0)
{
    using namespace SciCore;

    Real epsAbs = KCurrent.errorGoal();
    Real epsRel = 0;
    Real hMin   = defaultMinChebDistance(0.01, epsAbs);

    Model::SupervectorType rho0Vec = KCurrent.model()->vectorize(rho0);

    auto computeCurrent = [&](Real t) -> Real
    {
        Model::SupervectorType rhoVec_t = propagator.Pi()(t) * rho0Vec;

        Real safety           = 0.1;
        Complex timeLocalPart = KCurrent.SigmaInfty() * rhoVec_t;

        // In cases where the current kernel has narrow features, these could be missed
        // by the integration routine if we simply used
        //      integrateAdaptive(currentKernel.minusICurrentKernel(s) * rhoVec(t-s), 0, t)
        // Thus we set up integration sections based on the current kernel.
        RealVector sections = KCurrent.K().sections();
        int l               = 1;
        for (; l < sections.size(); ++l)
        {
            if (sections[l] >= t)
            {
                sections[l] = t;
                ++l;
                break;
            }
        }
        sections.conservativeResize(l);
        assert(sections.size() >= 2);
        assert(sections[0] == 0);
        assert(sections[sections.size() - 1] == t);

        return timeLocalPart.real() + integrateAdaptive(
                                          [&](Real s) -> Real
                                          {
                                              Model::SupervectorType rhoVec_t_minus_s =
                                                  propagator.Pi()(t - s) * rho0Vec;
                                              Complex returnComplex = KCurrent.K()(s) * rhoVec_t_minus_s;
                                              return returnComplex.real();
                                          },
                                          sections, safety* epsAbs, epsRel);
    };

    bool ok = false;
    ChebAdaptive<Real> returnValue(computeCurrent, KCurrent.K().sections(), epsAbs, epsRel, hMin, &ok);

    if (ok == false)
    {
        std::stringstream ss;
        ss << "Current accuracy goal not reached. Sections: " << returnValue.sections().transpose();
        throw Error(ss.str());
    }

    return returnValue;
}

} // namespace RealTimeTransport
