//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#define REAL_TIME_TRANSPORT_DEBUG

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>

#include <cereal/archives/binary.hpp>

#include <SciCore/Integration.h>

#include <RealTimeTransport/Models/AndersonDot.h>
#include <RealTimeTransport/Models/ResonantLevel.h>
#include <RealTimeTransport/RenormalizedPT/CurrentKernel.h>
#include <RealTimeTransport/RenormalizedPT/MemoryKernel.h>

using namespace SciCore;
using namespace RealTimeTransport;

//
// Functions related to exact solution resonant level model
//

// Eq. (C4) together with (C1) in J. Chem. Phys. 151, 044101
Real rlmExact_g(Real t, Real epsilon, Real mu, Real T, Real Gamma)
{
    using std::sin;
    using std::sinh;

    if (t == 0)
    {
        return 0;
    }

    auto integrand = [=](Real tau)
    {
        if (T != 0)
        {
            return sinh(Gamma / 2.0 * (t - tau)) / sinh(Gamma / 2.0 * t) *
                   (2.0 * T * sin((epsilon - mu) * tau) / sinh(M_PI * T * tau));
        }
        else
        {
            return sinh(Gamma / 2.0 * (t - tau)) / sinh(Gamma / 2.0 * t) *
                   (2.0 * sin((epsilon - mu) * tau) / (M_PI * tau));
        }
    };

    const Real epsAbs     = 1e-14;
    auto [result, absErr] = integrateAdaptive(integrand, 0, t, epsAbs, 0);
    assert(absErr < epsAbs);
    return result;
}

// Eq. (45) in J. Chem. Phys. 151, 044101
Real rlmExactParity(Real parity0, Real t, Real epsilon, Real mu, Real T, Real Gamma)
{
    using std::exp;
    using std::sin;
    using std::sinh;

    return parity0 * exp(-Gamma * t) + (1 - exp(-Gamma * t)) * rlmExact_g(t, epsilon, mu, T, Gamma);
}

StaticMatrix<4, 4> rlmExactMinusIK(
    Real t,
    Real epsilon,
    const RealVector& mu,
    const RealVector& T,
    const RealVector& Gamma)
{
    assert(mu.size() > 0);
    assert(mu.size() == T.size());
    assert(mu.size() == Gamma.size());

    Real prefactor = 0;
    Real pi        = std::numbers::pi_v<Real>;
    Real GammaSum  = Gamma.sum();

    if (t == 0)
    {
        for (int r = 0; r < mu.size(); ++r)
        {
            prefactor += 2 * Gamma[r] * ((epsilon - mu[r]) / pi);
        }
    }
    else
    {
        for (int r = 0; r < mu.size(); ++r)
        {
            if (T[r] == 0)
            {
                prefactor += 2 * Gamma[r] * (sin((epsilon - mu[r]) * t) / (pi * t)) * exp(-GammaSum / 2 * t);
            }
            else
            {
                prefactor +=
                    2 * Gamma[r] * (T[r] * sin((epsilon - mu[r]) * t) / sinh(pi * T[r] * t)) * exp(-GammaSum / 2 * t);
            }
        }
    }

    StaticMatrix<4, 4> Gpm{
        {0,  1, 0, 0}, //
        {0,  0, 0, 0}, //
        {1,  0, 0, 1}, //
        {0, -1, 0, 0}
    };
    Gpm /= std::sqrt(2);

    StaticMatrix<4, 4> Gpp{
        {0, 0, -1, 0}, //
        {1, 0,  0, 1}, //
        {0, 0,  0, 0}, //
        {0, 0,  1, 0}
    };
    Gpp /= std::sqrt(2);

    return prefactor * Gpm * Gpp;
}

StaticMatrix<2, 2> rlmExactT0StationaryState(Real epsilon, const RealVector& mu, const RealVector& Gamma)
{
    assert(mu.size() > 0);
    assert(mu.size() == Gamma.size());

    Real pi       = std::numbers::pi_v<Real>;
    Real GammaSum = Gamma.sum();

    Real p = 0;
    for (int r = 0; r < mu.size(); ++r)
    {
        p += 2 * Gamma[r] / (pi * GammaSum) * std::atan(2 / GammaSum * (epsilon - mu[r]));
    }

    StaticMatrix<2, 2> parity{
        {1,  0},
        {0, -1}
    };

    return 0.5 * StaticMatrix<2, 2>::Identity() + 0.5 * p * parity;
}

Real rlmExactT0Current(int r, Real epsilon, const RealVector& mu, const RealVector& Gamma)
{
    assert(mu.size() > 0);
    assert(mu.size() == Gamma.size());
    assert(r < mu.size());

    Real pi       = std::numbers::pi_v<Real>;
    Real GammaSum = Gamma.sum();

    Real returnValue = -Gamma[r] / pi * std::atan(2 / GammaSum * (epsilon - mu[r]));

    for (int s = 0; s < mu.size(); ++s)
    {
        returnValue += Gamma[r] / pi * Gamma[s] / GammaSum * std::atan(2 / GammaSum * (epsilon - mu[s]));
    }

    return returnValue;
}

//
// Functions related to exact solution of U=0 Anderson dot
//

enum class Spin
{
    Up,
    Down
};

// Wegewijs, Saptsov 2014 Eq. 134a
Real FPlus(Real t, Spin spin, int r, const AndersonDot& anderson)
{
    if (anderson.U() != 0)
    {
        throw Error("FPlus: only meaningful if U=0");
    }

    using std::exp;
    using std::sin;
    using std::sinh;

    Real spinEff         = (spin == Spin::Down) ? -1.0 : 1.0;
    Real epsilon_sigma_r = anderson.epsilon() + spinEff * anderson.B() / 2.0 - anderson.chemicalPotentials()[r];

    Real Gamma_sigma = (spin == Spin::Up) ? anderson.GammaUp().sum() : anderson.GammaDown().sum();

    Real T = anderson.temperatures()[r];
    for (int i = 0; i < anderson.numReservoirs(); ++i)
    {
        if (anderson.temperatures()[i] != T)
        {
            throw Error("FPlus: only meaningful if all temperatures are the same");
        }
    }

    return integrateAdaptive(
        [t, epsilon_sigma_r, Gamma_sigma, T](Real s)
        {
            if (s == 0)
            {
                return -epsilon_sigma_r / M_PI;
            }
            else
            {
                if (T == 0)
                {
                    return -sin(epsilon_sigma_r * s) / (M_PI * s) * exp(-0.5 * Gamma_sigma * s);
                }
                else
                {
                    return -T * sin(epsilon_sigma_r * s) / sinh(M_PI * T * s) * exp(-0.5 * Gamma_sigma * s);
                }
            }
        },
        0, t, 1e-15, 1e-15);
}

Real FMinus(Real t, Spin spin, int r, const AndersonDot& anderson)
{
    if (anderson.U() != 0)
    {
        throw Error("FMinus: only meaningful if U=0");
    }

    using std::exp;
    using std::sin;
    using std::sinh;

    Real spinEff         = (spin == Spin::Down) ? -1.0 : 1.0;
    Real epsilon_sigma_r = anderson.epsilon() + spinEff * anderson.B() / 2.0 - anderson.chemicalPotentials()[r];

    Real Gamma_sigma = (spin == Spin::Up) ? anderson.GammaUp().sum() : anderson.GammaDown().sum();

    Real T = anderson.temperatures()[r];
    for (int i = 0; i < anderson.numReservoirs(); ++i)
    {
        if (anderson.temperatures()[i] != T)
        {
            throw std::runtime_error("FMinus: only meaningful if all temperatures are the same");
        }
    }

    return integrateAdaptive(
        [t, epsilon_sigma_r, Gamma_sigma, T](Real s)
        {
            if (s == 0)
            {
                return +epsilon_sigma_r / M_PI * exp(-Gamma_sigma * t);
            }
            else
            {
                if (T == 0)
                {
                    return sin(epsilon_sigma_r * s) / (M_PI * s) * exp(-Gamma_sigma * t + 0.5 * Gamma_sigma * s);
                }
                else
                {
                    return T * sin(epsilon_sigma_r * s) / sinh(M_PI * T * s) *
                           exp(-Gamma_sigma * t + 0.5 * Gamma_sigma * s);
                }
            }
        },
        0, t, 1e-15, 1e-15);
}

// Wegewijs, Saptsov 2014 Eq. 143
Real Phi(Real t, Real Phi0, Spin spin, const AndersonDot& anderson)
{
    if (anderson.U() != 0)
    {
        throw std::runtime_error("Phi: only meaningful if U=0");
    }

    const RealVector& GammaSigmaVec = (spin == Spin::Up) ? anderson.GammaUp() : anderson.GammaDown();
    Real Gamma_sigma                = GammaSigmaVec.sum();

    Real sum = 0.0;
    for (int r = 0; r < anderson.numReservoirs(); ++r)
    {
        sum += GammaSigmaVec[r] / Gamma_sigma * (FPlus(t, spin, r, anderson) + FMinus(t, spin, r, anderson));
    }

    return sum + std::exp(-Gamma_sigma * t) * Phi0;
}

// Wegewijs, Saptsov 2014 Eq. 140b
Real theta(Real t, const AndersonDot& anderson)
{
    if (anderson.U() != 0)
    {
        throw Error("theta: only meaningful if U=0");
    }

    Real GammaUp  = anderson.GammaUp().sum();
    Real GammDown = anderson.GammaDown().sum();

    Real factor1 = 0;
    Real factor2 = 0;

    for (int r = 0; r < anderson.numReservoirs(); ++r)
    {
        factor1 +=
            anderson.GammaUp()[r] / GammaUp * (FPlus(t, Spin::Up, r, anderson) + FMinus(t, Spin::Up, r, anderson));

        factor2 += anderson.GammaDown()[r] / GammDown *
                   (FPlus(t, Spin::Down, r, anderson) + FMinus(t, Spin::Down, r, anderson));
    }

    return 4.0 * factor1 * factor2;
}

// Wegewijs, Saptsov 2014
Real nZeroInteraction(Real t, Real n0, Spin spin, const AndersonDot& anderson)
{
    if (anderson.U() != 0)
    {
        throw Error("nZeroInteraction: only meaningful if U=0");
    }

    const RealVector& GammaSigmaVec = (spin == Spin::Up) ? anderson.GammaUp() : anderson.GammaDown();
    Real Gamma_sigma                = GammaSigmaVec.sum();

    Real sum = 0;
    for (int r = 0; r < anderson.numReservoirs(); ++r)
    {
        sum += GammaSigmaVec[r] / Gamma_sigma * (FPlus(t, spin, r, anderson) + FMinus(t, spin, r, anderson));
    }

    return 0.5 + sum + std::exp(-Gamma_sigma * t) * (n0 - 0.5);
}

// Wegewijs, Saptsov 2014 Eq. 142 for initial states with vanishing total parity
Real totalParityZeroInteraction(Real t, Real nUp0, Real nDown0, const AndersonDot& anderson)
{
    if (anderson.U() != 0)
    {
        throw Error("totalParityZeroInteraction: only meaningful if U=0");
    }

    Real GammaUp   = anderson.GammaUp().sum();
    Real GammaDown = anderson.GammaDown().sum();

    Real sum = 0;
    for (int r = 0; r < anderson.numReservoirs(); ++r)
    {
        sum += exp(-GammaDown * t) * 4.0 * anderson.GammaUp()[r] / GammaUp *
                   (FPlus(t, Spin::Up, r, anderson) - FMinus(t, Spin::Up, r, anderson)) * (nDown0 - 0.5) +
               exp(-GammaUp * t) * 4.0 * anderson.GammaDown()[r] / GammaDown *
                   (FPlus(t, Spin::Down, r, anderson) - FMinus(t, Spin::Down, r, anderson)) * (nUp0 - 0.5);
    }

    return sum + theta(t, anderson);
}

// Wegewijs, Saptsov 2012 Eq. 252
Real stationaryCurrentZeroInteraction(const AndersonDot& anderson)
{
    if (anderson.U() != 0)
    {
        throw Error("stationaryCurrentZeroInteraction: only meaningful if U=0");
    }

    if (anderson.chemicalPotentials().size() != 2)
    {
        throw Error("stationaryCurrentZeroInteraction: needs exactly two reservoirs");
    }

    if (anderson.temperatures().array().abs().maxCoeff() != 0)
    {
        throw Error("stationaryCurrentZeroInteraction: needs zero temperature");
    }

    if (anderson.GammaUp() != anderson.GammaDown() || anderson.GammaUp()[0] != anderson.GammaUp()[1])
    {
        throw Error("stationaryCurrentZeroInteraction: all Γ_{rσ} must be the same");
    }

    Real Gamma   = anderson.GammaUp()[0];
    Real V       = anderson.chemicalPotentials()[0] - anderson.chemicalPotentials()[1];
    Real epsilon = anderson.epsilon();
    Real B       = anderson.B();

    Real returnValue = 0;
    for (int r = -1; r <= 1; r += 2)
    {
        for (int sigma = -1; sigma <= 1; sigma += 2)
        {
            returnValue += r * Gamma / (2 * M_PI) * std::atan((epsilon + sigma * B / 2 + r * V / 2) / Gamma);
        }
    }

    return returnValue;
}

TEST(FirstOrderExactSolutionRLM, FiniteTemperatureStationaryParity)
{
    Real epsilon = 2.5;
    RealVector T{{0.1}};
    RealVector mu{{0.2}};
    RealVector Gamma{{1}};
    auto model = createModel<ResonantLevel>(epsilon, T, mu, Gamma);

    std::vector errorGoals{1e-4, 1e-6, 1e-8, 1e-12, 1e-14};
    for (Real errorGoal : errorGoals)
    {
        Real tMax = 35;
        RenormalizedPT::MemoryKernel K(model, RenormalizedPT::Order::_1, tMax, errorGoal);

        Matrix rho  = K.stationaryState();
        Matrix rho2 = K.stationaryState(0); // Compute only using one block (other blocks are zero)
        EXPECT_ANY_THROW(K.stationaryState(1));

        EXPECT_TRUE(rho.isApprox(rho2));

        Real parity = (model->P() * rho).trace().real();
        Real exact  = rlmExactParity(0.0, 100.0, epsilon, mu[0], T[0], Gamma[0]);

        EXPECT_LT(std::abs(parity - exact), errorGoal) << "errorGoal = " << errorGoal;
    }
}

TEST(FirstOrderExactSolutionRLM, ZeroTemperatureStationaryQuantities)
{
    Real epsilon = 2.5;
    RealVector T{
        {0, 0, 0}
    };
    RealVector mu{
        {0.2, -4, 2}
    };
    RealVector Gamma{
        {1, 0.7, 1.6}
    };

    auto model = createModel<ResonantLevel>(epsilon, T, mu, Gamma);

    std::vector errorGoals{1e-4, 1e-6, 1e-8, 1e-12, 1e-14};
    for (Real errorGoal : errorGoals)
    {
        std::vector<int> blocks{-1, 0};
        for (int block : blocks)
        {
            Real tMax      = 20;
            auto K         = computeMemoryKernel(model, RenormalizedPT::Order::_1, tMax, errorGoal, block);
            auto KCurrent0 = computeCurrentKernel(model, 0, RenormalizedPT::Order::_1, tMax, errorGoal);
            auto KCurrent1 = computeCurrentKernel(model, 1, RenormalizedPT::Order::_1, tMax, errorGoal);
            auto KCurrent2 = computeCurrentKernel(model, 2, RenormalizedPT::Order::_1, tMax, errorGoal);

            // Check stationary state
            Matrix rhoStat  = K.stationaryState();
            Matrix rhoStat2 = K.stationaryState(0); // Compute only using one block (other blocks are zero)
            EXPECT_ANY_THROW(K.stationaryState(1));

            EXPECT_TRUE(rhoStat.isApprox(rhoStat2)) << "rhoStat =\n" << rhoStat << "\nrhoStat2 =\n" << rhoStat2;

            Matrix rhoStatExact = rlmExactT0StationaryState(epsilon, mu, Gamma);

            Real errAbs = maxNorm(rhoStat - rhoStatExact);
            EXPECT_LT(errAbs, errorGoal) << "rhoStat =\n" << rhoStat << "\nrhoStatExact =\n" << rhoStatExact;

            // Check stationary current
            Real stationaryCurrent0 = KCurrent0.stationaryCurrent(rhoStat);
            Real stationaryCurrent1 = KCurrent1.stationaryCurrent(rhoStat);
            Real stationaryCurrent2 = KCurrent2.stationaryCurrent(rhoStat);

            EXPECT_LT(std::abs(stationaryCurrent0 - rlmExactT0Current(0, epsilon, mu, Gamma)), errorGoal)
                << "computed current = " << stationaryCurrent0
                << "\nexpected current = " << rlmExactT0Current(0, epsilon, mu, Gamma) << "\nerrorGoal = " << errorGoal;
            EXPECT_LT(std::abs(stationaryCurrent1 - rlmExactT0Current(1, epsilon, mu, Gamma)), errorGoal)
                << "computed current = " << stationaryCurrent1
                << "\nexpected current = " << rlmExactT0Current(1, epsilon, mu, Gamma) << "\nerrorGoal = " << errorGoal;
            EXPECT_LT(std::abs(stationaryCurrent2 - rlmExactT0Current(2, epsilon, mu, Gamma)), errorGoal)
                << "computed current = " << stationaryCurrent2
                << "\nexpected current = " << rlmExactT0Current(2, epsilon, mu, Gamma) << "\nerrorGoal = " << errorGoal;
        }
    }
}

TEST(FirstOrderExactSolutionRLM, computeTransientParity)
{
    Real epsilon = 2.5;
    RealVector T{{0.1}};
    RealVector mu{{0.2}};
    RealVector Gamma{{1}};

    auto model = createModel<ResonantLevel>(epsilon, T, mu, Gamma);

    Real tMax = 7;
    Matrix rho0{
        {0.3,   0},
        {  0, 0.7}
    };

    Real parity0       = (model->P() * rho0).trace().real();
    RealVector tValues = RealVector::LinSpaced(500, 0.0, tMax);

    std::vector errorGoals{1e-4, 1e-6, 1e-8, 1e-12, 1e-14};
    for (Real errorGoal : errorGoals)
    {
        std::vector<int> blocks{-1, 0};
        for (int block : blocks)
        {
            auto K = computeMemoryKernel(model, RenormalizedPT::Order::_1, tMax, errorGoal, block);

            int blockIndex = 0;
            auto PiBlock0  = computePropagator(K, blockIndex);
            auto Pi        = computePropagator(K);

            for (Real t : tValues)
            {
                Matrix rho_t      = PiBlock0(t, rho0);
                Matrix rho_t_full = Pi(t, rho0);

                EXPECT_TRUE(rho_t_full.isApprox(rho_t));

                Complex cplxParity = (model->P() * rho_t).trace();
                EXPECT_LT(maxNorm(cplxParity.imag()), 1e-15);

                Real parity = cplxParity.real();
                Real exact  = rlmExactParity(parity0, t, epsilon, mu[0], T[0], Gamma[0]);
                EXPECT_LT(maxNorm(parity - exact), errorGoal)
                    << "errorGoal = " << errorGoal << "t = " << t << "parity = " << parity;
            }
        }
    }
}

TEST(SecondOrderExactSolutionAndersonDot, TransientOccupation)
{
    struct TestCase
    {
        Real epsilon = 0;
        Real B       = 0;
        Real T       = 0;
        RealVector mu{
            {0, 0}
        };
        RealVector Gamma{
            {1.0, 1.0}
        };
        Real tMax      = 6;
        Real errorGoal = 1e-6;
    };

    // clang-format off
    std::vector<TestCase> testCases{
        {.epsilon =  0, .B = 0,   .T = 0.2, .mu = RealVector{{5, -5}}    , .Gamma = RealVector{{1.0, 1.0}}, .tMax = 8, .errorGoal = 1e-12},
        {.epsilon =  5, .B = 0,   .T = 0.0, .mu = RealVector{{-0.1, 0.1}}, .Gamma = RealVector{{1.0, 1.0}}, .tMax = 6, .errorGoal = 1e-6},
        {.epsilon =  2, .B = 1,   .T = 0.1, .mu = RealVector{{-0.3, 0.1}}, .Gamma = RealVector{{0.9, 1.3}}, .tMax = 8, .errorGoal = 1e-8},
        {.epsilon =  6, .B = 1,   .T = 0.1, .mu = RealVector{{-0.3, 0.1}}, .Gamma = RealVector{{0.9, 1.3}}, .tMax = 8, .errorGoal = 1e-8},
        {.epsilon =  6, .B = 1,   .T = 0.1, .mu = RealVector{{-0.3, 0.1}}, .Gamma = RealVector{{0.9, 1.3}}, .tMax = 8, .errorGoal = 1e-10},
        {.epsilon =  0, .B = 0.1, .T = 0.1, .mu = RealVector{{-0.3, 0.1}}, .Gamma = RealVector{{0.9, 1.3}}, .tMax = 8, .errorGoal = 1e-10},
        {.epsilon =  6, .B = 1,   .T = 0.0, .mu = RealVector{{-0.3, 0.1}}, .Gamma = RealVector{{0.9, 1.3}}, .tMax = 8, .errorGoal = 1e-12},
        {.epsilon =  20, .B = 5,  .T = 2.0, .mu = RealVector{{-10, 10}},   .Gamma = RealVector{{0.9, 1.3}}, .tMax = 20, .errorGoal = 1e-4},
        {.epsilon = -75, .B = 30, .T = 2.0, .mu = RealVector{{-20, 21}},   .Gamma = RealVector{{0.9, 1.3}}, .tMax = 5, .errorGoal = 1e-4}
        };
    // clang-format on

    Matrix occupationUp{
        {0, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 1}
    };

    Matrix occupationDown{
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };

    Matrix rho0{
        {0.25,   0,   0,    0},
        {   0, 0.1,   0,    0},
        {   0,   0, 0.5,    0},
        {   0,   0,   0, 0.15}
    };

    Real nUp0   = (rho0 * occupationUp).trace().real();
    Real nDown0 = (rho0 * occupationDown).trace().real();

    tf::Executor executor;
    for (const auto& test : testCases)
    {
        Real U       = 0;
        RealVector T = RealVector::Constant(test.mu.size(), test.T);

        auto model = createModel<AndersonDot>(test.epsilon, test.B, U, T, test.mu, test.Gamma);

        std::vector<int> blocks{-1, 0};
        for (int block : blocks)
        {
            auto start = std::chrono::steady_clock::now();
            auto K = computeMemoryKernel(model, RenormalizedPT::Order::_2, test.tMax, test.errorGoal, executor, block);
            auto stop     = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "Computed memory kernel in: " << duration.count() << " milliseconds" << std::endl;

            auto Pi = computePropagator(K);

            RealVector tValues = RealVector::LinSpaced(500, 0, test.tMax);
            for (Real t : tValues)
            {
                Real nUpExact   = nZeroInteraction(t, nUp0, Spin::Up, dynamic_cast<AndersonDot&>(*model));
                Real nDownExact = nZeroInteraction(t, nDown0, Spin::Down, dynamic_cast<AndersonDot&>(*model));

                Matrix rho_t = Pi(t, rho0);

                Real nUpComputed1   = (occupationUp * rho_t).trace().real();
                Real nDownComputed1 = (occupationDown * rho_t).trace().real();

                EXPECT_LT(std::abs(nUpComputed1 - nUpExact), 4 * test.errorGoal)
                    << "t = " << t << " nUpComputed1 = " << nUpComputed1 << " nUpExact = " << nUpExact;
                EXPECT_LT(std::abs(nDownComputed1 - nDownExact), 4 * test.errorGoal)
                    << "t = " << t << " nDownComputed1 = " << nDownComputed1 << " nDownExact = " << nDownExact;
            }
        }
    }
}

TEST(SecondOrderAndersonDot, DoesItFinish)
{
    // clang-format off
    // These test cases were previously buggy and didn't finish / threw an exception
    struct TestCase
    {
        Real epsilon = 0;
        Real U       = 0;
        Real B       = 0;
        Real T       = 0;
        RealVector mu{{0, 0}};
        RealVector Gamma{{1.0, 1.0}};
        Real tMax      = 6;
        Real errorGoal = 1e-6;
        int block      = -1;
        Real hMin      = 1e-4;
    };

    
    std::vector<TestCase> testCases{
        {.epsilon = -15, .U = 30, .B = 5, .T = 0, .mu = RealVector{{8, -8}}, .Gamma = RealVector{{1.0, 1.0}}, .tMax = 20, .errorGoal = 1e-7, .block = 3, .hMin = 0.5},
        };
    // clang-format on

    for (const auto& test : testCases)
    {
        RealVector T = RealVector::Constant(test.mu.size(), test.T);

        auto model = createModel<AndersonDot>(test.epsilon, test.B, test.U, T, test.mu, test.Gamma);
        RenormalizedPT::MemoryKernel K;
        EXPECT_NO_THROW(K.initialize(
            model.get(), RenormalizedPT::Order::_2, test.tMax, test.errorGoal, nullptr, test.block, test.hMin,
            nullptr));
    }
}

TEST(AndersonDot, ZeroTemperatureStationaryCurrent)
{
    Real U = 0;
    RealVector T{
        {0, 0}
    };
    RealVector mu{
        {5, -5}
    };
    Real B       = 1;
    Real epsilon = 2;
    RealVector Gamma{
        {2, 2}
    };
    Real tMax      = 20;
    Real errorGoal = 1e-8;

    auto model = createModel<AndersonDot>(epsilon, B, U, T, mu, Gamma);

    auto start    = std::chrono::steady_clock::now();
    auto K        = computeMemoryKernel(model, RenormalizedPT::Order::_2, tMax, errorGoal);
    auto stop     = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Computed memory kernel in: " << duration.count() << " milliseconds" << std::endl;
    Matrix rho = K.stationaryState();

    // First order is already exact
    {
        start         = std::chrono::steady_clock::now();
        auto KCurrent = computeCurrentKernel(model, 0, RenormalizedPT::Order::_1, tMax, errorGoal);
        stop          = std::chrono::steady_clock::now();
        duration      = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Computed current kernel in: " << duration.count() << " milliseconds" << std::endl;

        Real statCurrent  = KCurrent.stationaryCurrent(rho);
        Real exactCurrent = stationaryCurrentZeroInteraction(dynamic_cast<AndersonDot&>(*model));

        EXPECT_LT(std::abs(statCurrent - exactCurrent), errorGoal)
            << "statCurrent = " << statCurrent << " exactCurrent = " << exactCurrent;

        // Check that transient current at tMax is equal to stationary current
        Model::OperatorType rho0 = Model::OperatorType::Zero(4, 4);
        rho0(0, 0)               = 1;

        auto Pi      = computePropagator(K);
        auto current = computeCurrent(KCurrent, Pi, rho0);
        EXPECT_LT(std::abs(current(tMax) - exactCurrent), errorGoal)
            << "current(tMax) = " << current(tMax) << " exactCurrent = " << exactCurrent;
    }

    // Check that second order doesn't break anything
    {
        start         = std::chrono::steady_clock::now();
        auto KCurrent = computeCurrentKernel(model, 0, RenormalizedPT::Order::_2, tMax, errorGoal);
        stop          = std::chrono::steady_clock::now();
        duration      = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Computed current kernel in: " << duration.count() << " milliseconds" << std::endl;

        Real current      = KCurrent.stationaryCurrent(rho);
        Real exactCurrent = stationaryCurrentZeroInteraction(dynamic_cast<AndersonDot&>(*model));

        EXPECT_LT(std::abs(current - exactCurrent), errorGoal)
            << "current = " << current << " exactCurrent = " << exactCurrent;
    }
}

TEST(MemoryKernel, Serialization)
{
    Real epsilon = 2.5;
    RealVector T{
        {0.1, 10, 4}
    };
    RealVector mu{
        {0.2, -3, 5}
    };
    RealVector Gamma{
        {1, 2, 3}
    };
    auto model = createModel<ResonantLevel>(epsilon, T, mu, Gamma);

    Real errorGoal = 1e-6;
    Real tMax      = 35;
    RenormalizedPT::MemoryKernel K(model, RenormalizedPT::Order::_1, tMax, errorGoal);

    std::string archiveFilename = "renormalized_pt_memory_kernel_test_out.cereal";
    std::remove(archiveFilename.c_str());

    {
        std::ofstream os(archiveFilename, std::ios::binary);
        cereal::BinaryOutputArchive archive(os);
        archive(K);
    }

    {
        RenormalizedPT::MemoryKernel fromFile;
        {
            std::ifstream is(archiveFilename, std::ios::binary);
            cereal::BinaryInputArchive archive(is);
            archive(fromFile);
        }

        EXPECT_EQ(dynamic_cast<const ResonantLevel*>(fromFile.model())->epsilon(), epsilon);
        EXPECT_EQ(dynamic_cast<const ResonantLevel*>(fromFile.model())->temperatures(), T);
        EXPECT_EQ(dynamic_cast<const ResonantLevel*>(fromFile.model())->chemicalPotentials(), mu);
        EXPECT_EQ(*fromFile.model(), *model);
        EXPECT_EQ(K, fromFile);
    }
}
