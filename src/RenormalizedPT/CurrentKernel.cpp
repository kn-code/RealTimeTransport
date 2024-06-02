//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/RenormalizedPT/CurrentKernel.h"

#include <sstream>

#include <SciCore/Integration.h>

#include "RealTimeTransport/BlockMatrices/MatrixExp.h"
#include "RealTimeTransport/BlockMatrices/MatrixOperations.h"
#include "RealTimeTransport/RenormalizedPT/Diagrams.h"
#include "RealTimeTransport/Utility.h"

namespace RealTimeTransport::RenormalizedPT
{

CurrentKernel::CurrentKernel() noexcept
{
}

CurrentKernel::CurrentKernel(CurrentKernel&& other) noexcept
    : _model(std::move(other._model)), _errorGoal(other._errorGoal),
      _minusISigmaInfty(std::move(other._minusISigmaInfty)), _minusIK(std::move(other._minusIK))
{
}

CurrentKernel::CurrentKernel(const CurrentKernel& other)
    : _model(nullptr), _errorGoal(other._errorGoal), _minusISigmaInfty(other._minusISigmaInfty),
      _minusIK(other._minusIK)
{
    if (other._model.get() != nullptr)
    {
        _model = other._model->copy();
    }
}

CurrentKernel& CurrentKernel::operator=(CurrentKernel&& other)
{
    _model            = std::move(other._model);
    _errorGoal        = other._errorGoal;
    _minusISigmaInfty = std::move(other._minusISigmaInfty);
    _minusIK          = std::move(other._minusIK);

    return *this;
}

CurrentKernel& CurrentKernel::operator=(const CurrentKernel& other)
{
    if (other._model.get() != nullptr)
    {
        _model = other._model->copy();
    }
    else
    {
        _model.reset();
    }

    _errorGoal        = other._errorGoal;
    _minusISigmaInfty = other._minusISigmaInfty;
    _minusIK          = other._minusIK;

    return *this;
}

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
    const Model* model,
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

    _model     = model->copy();
    _errorGoal = errorGoal;

    // Compute superfermions
    auto superfermion             = computeAllSuperfermions(Keldysh::Plus, model);
    auto superfermionAnnihilation = computeAllSuperfermions(Keldysh::Minus, model);

    // Compute infinite temperature Liouvillian and current kernel
    BlockDiagonalMatrix minusILInfty =
        computeLiouvillian(model) + computeSigmaInfty(superfermion, superfermionAnnihilation, model);
    _minusISigmaInfty = computeSigmaInftyCurrent(r, superfermionAnnihilation, model);
    BlockDiagonalMatrixExp piInfty(minusILInfty);
    std::function<BlockDiagonalMatrix(Real)> computePiInfty = [&](Real t) -> BlockDiagonalMatrix
    {
        return piInfty(t);
    };

    std::function<BlockDiagonalMatrix(Real)> computePiInftyM1 = [&](Real t) -> BlockDiagonalMatrix
    {
        return piInfty.expm1(t);
    };

    // Compute the trace functional as rowvector (idRow)
    int dim              = model->dimHilbertSpace();
    Operator id          = Operator::Identity(dim, dim);
    Supervector idCol    = model->vectorize(id);
    SuperRowVector idRow = idCol.transpose();

    // Set up information about block structure
    const std::vector<int>& blockDims = model->blockDimensions();

    int numBlocks = blockDims.size();
    std::vector<int> blockStartIndices(numBlocks, 0);
    for (int i = 1; i < numBlocks; ++i)
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

    Real tCrit          = std::min(2 / maxNorm(minusILInfty), tMax);
    RealVector sections = defaultInitialChebSections(tMax, tCrit);
    std::vector<RealVector> allInitialSections(numBlocks, sections);

    Real epsAbs = errorGoal;
    Real epsRel = 0;
    Real safety = 0.1;
    Real hMin   = defaultMinChebDistance(tCrit, errorGoal);

    Error accuracyError;
    bool hasAccuracyError = false;

    auto diagram_1 = [&](Real t) -> SuperRowVector
    {
        return Detail::currentDiagram_1(
            t, r, tCrit, computePiInfty, computePiInftyM1, idRow, Tr_superfermionAnnihilation, superfermion,
            blockStartIndices, model);
    };

    if (order == Order::_1)
    {
        bool ok = false;
        if (executor == nullptr)
        {
            _minusIK = ChebAdaptive<SuperRowVector>(diagram_1, sections, epsAbs, epsRel, hMin, &ok);
        }
        else
        {
            _minusIK = ChebAdaptive<SuperRowVector>(diagram_1, sections, epsAbs, epsRel, hMin, *executor, &ok);
        }

        if (ok == false && hasAccuracyError == false)
        {
            hasAccuracyError = true;
            std::stringstream ss;
            ss << "First order renormalized current kernel computation failed. Sections = "
               << _minusIK.sections().transpose();
            accuracyError = Error(ss.str());
        }
    }
    else if (order == Order::_2)
    {
        bool ok = false;

        BlockDiagonalCheb memoryKernelDiagram_1;
        if (executor == nullptr)
        {
            memoryKernelDiagram_1 = BlockDiagonalCheb(
                [&](int blockIndex, Real t) -> Matrix {
                    return Detail::diagram_1(
                        blockIndex, t, tCrit, computePiInfty, computePiInftyM1, superfermion, model);
                },
                numBlocks, allInitialSections, safety* epsAbs, epsRel, hMin, &ok);
        }
        else
        {
            memoryKernelDiagram_1 = BlockDiagonalCheb(
                [&](int blockIndex, Real t) -> Matrix {
                    return Detail::diagram_1(
                        blockIndex, t, tCrit, computePiInfty, computePiInftyM1, superfermion, model);
                },
                numBlocks, allInitialSections, safety* epsAbs, epsRel, hMin, *executor, &ok);
        }

        if (ok == false && hasAccuracyError == false)
        {
            hasAccuracyError = true;
            std::stringstream ss;
            ss << "Second order renormalized current kernel computation failed [memoryKernelDiagram_1]. Sections:\n";
            for (const auto& sec : memoryKernelDiagram_1.sections())
            {
                ss << sec.transpose() << "\n";
            }
            accuracyError = Error(ss.str());
        }

        ChebAdaptive<SuperRowVector> chebDiagram1;
        if (executor == nullptr)
        {
            chebDiagram1 = ChebAdaptive<SuperRowVector>(diagram_1, sections, safety * epsAbs, epsRel, hMin, &ok);
        }
        else
        {
            chebDiagram1 =
                ChebAdaptive<SuperRowVector>(diagram_1, sections, safety * epsAbs, epsRel, hMin, *executor, &ok);
        }

        if (ok == false && hasAccuracyError == false)
        {
            hasAccuracyError = true;
            std::stringstream ss;
            ss << "Second order renormalized current kernel computation failed [chebDiagram1]. Sections: "
               << chebDiagram1.sections().transpose();
            accuracyError = Error(ss.str());
        }

        std::function<BlockVector(int, int, Real, Real)> computeD = [&](int i, int col, Real t, Real s) -> BlockVector
        {
            return Detail::effectiveVertexDiagram1_col(
                i, col, t, s, tCrit, safety * epsAbs, computePiInfty, computePiInftyM1, superfermion, model);
        };

        auto diagram_2_1 = [&](Real t) -> SuperRowVector
        {
            return Detail::currentDiagram_2(
                t, r, safety * epsAbs, epsRel, computePiInfty, computeD, Tr_superfermionAnnihilation, blockStartIndices,
                block, model);
        };

        auto diagram_2_2 = [&](Real t) -> SuperRowVector
        {
            return Detail::currentDiagram_2_2(
                t, r, safety * epsAbs, epsRel, computePiInfty, memoryKernelDiagram_1, superfermion,
                Tr_superfermionAnnihilation, blockStartIndices, model);
        };

        if (executor == nullptr)
        {
            _minusIK = ChebAdaptive<SuperRowVector>(
                [&](Real t) -> SuperRowVector { return chebDiagram1(t) + diagram_2_1(t) + diagram_2_2(t); },
                chebDiagram1.sections(), epsAbs, epsRel, hMin, &ok);
        }
        else
        {
            _minusIK = ChebAdaptive<SuperRowVector>(
                [&](Real t) -> SuperRowVector { return chebDiagram1(t) + diagram_2_1(t) + diagram_2_2(t); },
                chebDiagram1.sections(), epsAbs, epsRel, hMin, *executor, &ok);
        }

        if (ok == false && hasAccuracyError == false)
        {
            hasAccuracyError = true;
            std::stringstream ss;
            ss << "Second order renormalized current kernel computation failed. Sections: "
               << _minusIK.sections().transpose();
            accuracyError = Error(ss.str());
        }
    }
    else
    {
        throw Error("Not implemented");
    }

    if (hasAccuracyError == true)
    {
        CurrentKernel preliminaryResult;
        preliminaryResult._model            = std::move(_model);
        preliminaryResult._errorGoal        = _errorGoal;
        preliminaryResult._minusISigmaInfty = std::move(_minusISigmaInfty);
        preliminaryResult._minusIK          = std::move(_minusIK);

        throw AccuracyError<CurrentKernel>(std::move(accuracyError), std::move(preliminaryResult));
    }
}

} // namespace RealTimeTransport::RenormalizedPT

namespace RealTimeTransport
{

SciCore::ChebAdaptive<SciCore::Real> computeCurrent(
    const RenormalizedPT::CurrentKernel& KCurrent,
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
        ss << "Compute current accuracy goal not reached. Sections: " << returnValue.sections().transpose();
        throw AccuracyError<ChebAdaptive<Real>>(ss.str(), std::move(returnValue));
    }

    return returnValue;
}

} // namespace RealTimeTransport
