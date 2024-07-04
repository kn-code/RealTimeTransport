//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include <gtest/gtest.h>

#include <RealTimeTransport/Models/AndersonDot.h>
#include <RealTimeTransport/RenormalizedPT/ConductanceKernel.h>

using namespace SciCore;
using namespace RealTimeTransport;

// clang-format off
struct TestCase
{
    Real epsilon;
    Real B;
    Real U;
    RealVector T;
    RealVector mu;
    Real tMax;
    Real errorGoalLow;
    Real errorGoalHigh;
    int block;
    RealVector Gamma{{1.1, 0.8}};
};
// clang-format on

TEST(ConductanceKernel, firstOrderConductance)
{
    // clang-format off
    std::vector<TestCase> testCases{
        {.epsilon = -4, .B = 0.5, .U = 10, .T = RealVector{{0.2, 0.3}}, .mu = RealVector{{0.0,  0.0}}, .tMax = 30, .errorGoalLow = 1e-6, .errorGoalHigh = 1e-14, .block = 0},
        {.epsilon = -4, .B = 0.5, .U = 10, .T = RealVector{{0.2, 0.3}}, .mu = RealVector{{0.0,  0.0}}, .tMax = 30, .errorGoalLow = 1e-6, .errorGoalHigh = 1e-14, .block = -1},
        {.epsilon = -4, .B = 0.5, .U = 10, .T = RealVector{{0.2, 0.3}}, .mu = RealVector{{5.6, -4.0}}, .tMax = 30, .errorGoalLow = 1e-6, .errorGoalHigh = 1e-14, .block = 0},
        {.epsilon = -4, .B = 0.5, .U = 10, .T = RealVector{{0.2, 0.3}}, .mu = RealVector{{5.6, -4.0}}, .tMax = 30, .errorGoalLow = 1e-6, .errorGoalHigh = 1e-14, .block = -1},
        {.epsilon = 10, .B = 3.5, .U = 10, .T = RealVector{{0.2, 0.3}}, .mu = RealVector{{1.6, -1.0}}, .tMax = 30, .errorGoalLow = 1e-6, .errorGoalHigh = 1e-14, .block = 0},
        {.epsilon = 10, .B = 3.5, .U = 10, .T = RealVector{{0.2, 0.3}}, .mu = RealVector{{1.6, -1.0}}, .tMax = 30, .errorGoalLow = 1e-6, .errorGoalHigh = 1e-14, .block = -1}
    };
    // clang-format on

    for (const auto& test : testCases)
    {
        int r = 0;

        // Compute with lower accuracy conductance kernel
        auto method = RenormalizedPT::Order::_1;
        auto model  = createModel<AndersonDot>(test.epsilon, test.B, test.U, test.T, test.mu, test.Gamma);
        auto K      = computeMemoryKernel(model, method, test.tMax, test.errorGoalLow, test.block);
        auto KI     = computeCurrentKernel(model, r, method, test.tMax, test.errorGoalLow, test.block);
        auto KC     = computeConductanceKernel(K, KI, method, test.block);

        // Compute with higher accuracy and numerical differentiation
        Real dmu        = 10 * std::sqrt(test.errorGoalHigh);
        RealVector mu1  = test.mu;
        mu1[r]         += dmu / 2;
        RealVector mu2  = test.mu;
        mu2[r]         -= dmu / 2;

        auto model1 = createModel<AndersonDot>(test.epsilon, test.B, test.U, test.T, mu1, test.Gamma);
        auto K1     = computeMemoryKernel(model1, method, test.tMax, test.errorGoalHigh, test.block);
        auto KI1    = computeCurrentKernel(model1, r, method, test.tMax, test.errorGoalHigh, test.block);

        auto model2 = createModel<AndersonDot>(test.epsilon, test.B, test.U, test.T, mu2, test.Gamma);
        auto K2     = computeMemoryKernel(model2, method, test.tMax, test.errorGoalHigh, test.block);
        auto KI2    = computeCurrentKernel(model2, r, method, test.tMax, test.errorGoalHigh, test.block);

        Real I1    = KI1.stationaryCurrent(K1.stationaryState());
        Real I2    = KI2.stationaryCurrent(K2.stationaryState());
        Real dIdmu = (I1 - I2) / dmu;

        EXPECT_LE(std::abs(dIdmu - KC.conductance()), test.errorGoalLow)
            << "I1=" << I1 << " I2=" << I2 << " dIdmu=" << dIdmu << " KC.conductance()=" << KC.conductance();
    }
}

TEST(ConductanceKernel, firstOrderCrossConductance)
{
    // clang-format off
    std::vector<TestCase> testCases{
        {.epsilon = -4, .B = 0.5, .U = 10, .T = RealVector{{0.2, 0.3}}, .mu = RealVector{{0.0,  0.0}}, .tMax = 30, .errorGoalLow = 1e-6, .errorGoalHigh = 1e-14, .block = 0},
        {.epsilon = -4, .B = 0.5, .U = 10, .T = RealVector{{0.2, 0.3}}, .mu = RealVector{{0.0,  0.0}}, .tMax = 30, .errorGoalLow = 1e-6, .errorGoalHigh = 1e-14, .block = -1},
        {.epsilon = -4, .B = 0.5, .U = 10, .T = RealVector{{0.2, 0.3}}, .mu = RealVector{{5.6, -4.0}}, .tMax = 30, .errorGoalLow = 1e-6, .errorGoalHigh = 1e-14, .block = 0},
        {.epsilon = -4, .B = 0.5, .U = 10, .T = RealVector{{0.2, 0.3}}, .mu = RealVector{{5.6, -4.0}}, .tMax = 30, .errorGoalLow = 1e-6, .errorGoalHigh = 1e-14, .block = -1},
        {.epsilon = 10, .B = 3.5, .U = 10, .T = RealVector{{0.2, 0.3}}, .mu = RealVector{{1.6, -1.0}}, .tMax = 30, .errorGoalLow = 1e-6, .errorGoalHigh = 1e-14, .block = 0},
        {.epsilon = 10, .B = 3.5, .U = 10, .T = RealVector{{0.2, 0.3}}, .mu = RealVector{{1.6, -1.0}}, .tMax = 30, .errorGoalLow = 1e-6, .errorGoalHigh = 1e-14, .block = -1}
    };
    // clang-format on

    for (const auto& test : testCases)
    {
        int rI  = 0;
        int rmu = 1;

        // Compute with lower accuracy conductance kernel
        auto method = RenormalizedPT::Order::_1;
        auto model  = createModel<AndersonDot>(test.epsilon, test.B, test.U, test.T, test.mu, test.Gamma);
        auto K      = computeMemoryKernel(model, method, test.tMax, test.errorGoalLow, test.block);
        auto KI     = computeCurrentKernel(model, rI, method, test.tMax, test.errorGoalLow, test.block);
        auto KC     = computeConductanceKernel(K, KI, rmu, method, test.block);

        // Compute with higher accuracy and numerical differentiation
        Real dmu        = 10 * std::sqrt(test.errorGoalHigh);
        RealVector mu1  = test.mu;
        mu1[rmu]       += dmu / 2;
        RealVector mu2  = test.mu;
        mu2[rmu]       -= dmu / 2;

        auto model1 = createModel<AndersonDot>(test.epsilon, test.B, test.U, test.T, mu1, test.Gamma);
        auto K1     = computeMemoryKernel(model1, method, test.tMax, test.errorGoalHigh, test.block);
        auto KI1    = computeCurrentKernel(model1, rI, method, test.tMax, test.errorGoalHigh, test.block);

        auto model2 = createModel<AndersonDot>(test.epsilon, test.B, test.U, test.T, mu2, test.Gamma);
        auto K2     = computeMemoryKernel(model2, method, test.tMax, test.errorGoalHigh, test.block);
        auto KI2    = computeCurrentKernel(model2, rI, method, test.tMax, test.errorGoalHigh, test.block);

        Real I1    = KI1.stationaryCurrent(K1.stationaryState());
        Real I2    = KI2.stationaryCurrent(K2.stationaryState());
        Real dIdmu = (I1 - I2) / dmu;

        EXPECT_LE(std::abs(dIdmu - KC.conductance()), test.errorGoalLow)
            << "I1=" << I1 << " I2=" << I2 << " dIdmu=" << dIdmu << " KC.conductance()=" << KC.conductance();
    }
}

TEST(ConductanceKernel, secondOrderConductance)
{
    // clang-format off
    std::vector<TestCase> testCases{
        {.epsilon = -4, .B = 0.5, .U = 10, .T = RealVector{{0.2, 0.3}}, .mu = RealVector{{0.0,  0.0}}, .tMax = 20, .errorGoalLow = 1e-4, .errorGoalHigh = 1e-6, .block = 0},
        {.epsilon = -4, .B = 0.5, .U = 10, .T = RealVector{{0.2, 0.3}}, .mu = RealVector{{5.6, -4.0}}, .tMax = 20, .errorGoalLow = 1e-4, .errorGoalHigh = 1e-6, .block = -1}
    };
    // clang-format on

    for (const auto& test : testCases)
    {
        int r = 0;

        // Compute with lower accuracy conductance kernel
        auto method = RenormalizedPT::Order::_2;
        auto model  = createModel<AndersonDot>(test.epsilon, test.B, test.U, test.T, test.mu, test.Gamma);
        auto K      = computeMemoryKernel(model, method, test.tMax, test.errorGoalLow, test.block);
        auto KI     = computeCurrentKernel(model, r, method, test.tMax, test.errorGoalLow, test.block);
        auto KC     = computeConductanceKernel(K, KI, method, test.block);

        // Compute with higher accuracy and numerical differentiation
        Real dmu        = 10 * std::sqrt(test.errorGoalHigh);
        RealVector mu1  = test.mu;
        mu1[r]         += dmu / 2;
        RealVector mu2  = test.mu;
        mu2[r]         -= dmu / 2;

        auto model1 = createModel<AndersonDot>(test.epsilon, test.B, test.U, test.T, mu1, test.Gamma);
        auto K1     = computeMemoryKernel(model1, method, test.tMax, test.errorGoalHigh, test.block);
        auto KI1    = computeCurrentKernel(model1, r, method, test.tMax, test.errorGoalHigh, test.block);

        auto model2 = createModel<AndersonDot>(test.epsilon, test.B, test.U, test.T, mu2, test.Gamma);
        auto K2     = computeMemoryKernel(model2, method, test.tMax, test.errorGoalHigh, test.block);
        auto KI2    = computeCurrentKernel(model2, r, method, test.tMax, test.errorGoalHigh, test.block);

        Real I1    = KI1.stationaryCurrent(K1.stationaryState());
        Real I2    = KI2.stationaryCurrent(K2.stationaryState());
        Real dIdmu = (I1 - I2) / dmu;

        EXPECT_LE(std::abs(dIdmu - KC.conductance()), test.errorGoalLow)
            << "I1=" << I1 << " I2=" << I2 << " dIdmu=" << dIdmu << " KC.conductance()=" << KC.conductance();
    }
}

TEST(ConductanceKernel, secondOrderCrossConductance)
{
    // clang-format off
    std::vector<TestCase> testCases{
        {.epsilon = -4, .B = 0.5, .U = 10, .T = RealVector{{0.2, 0.3}}, .mu = RealVector{{0.0,  0.0}}, .tMax = 20, .errorGoalLow = 1e-4, .errorGoalHigh = 1e-6, .block = 0},
        {.epsilon = -4, .B = 0.5, .U = 10, .T = RealVector{{0.2, 0.3}}, .mu = RealVector{{5.6, -4.0}}, .tMax = 20, .errorGoalLow = 1e-4, .errorGoalHigh = 1e-6, .block = -1}
    };
    // clang-format on

    for (const auto& test : testCases)
    {
        int rI  = 0;
        int rmu = 1;

        // Compute with lower accuracy conductance kernel
        auto method = RenormalizedPT::Order::_2;
        auto model  = createModel<AndersonDot>(test.epsilon, test.B, test.U, test.T, test.mu, test.Gamma);
        auto K      = computeMemoryKernel(model, method, test.tMax, test.errorGoalLow, test.block);
        auto KI     = computeCurrentKernel(model, rI, method, test.tMax, test.errorGoalLow, test.block);
        auto KC     = computeConductanceKernel(K, KI, rmu, method, test.block);

        // Compute with higher accuracy and numerical differentiation
        Real dmu        = 10 * std::sqrt(test.errorGoalHigh);
        RealVector mu1  = test.mu;
        mu1[rmu]       += dmu / 2;
        RealVector mu2  = test.mu;
        mu2[rmu]       -= dmu / 2;

        auto model1 = createModel<AndersonDot>(test.epsilon, test.B, test.U, test.T, mu1, test.Gamma);
        auto K1     = computeMemoryKernel(model1, method, test.tMax, test.errorGoalHigh, test.block);
        auto KI1    = computeCurrentKernel(model1, rI, method, test.tMax, test.errorGoalHigh, test.block);

        auto model2 = createModel<AndersonDot>(test.epsilon, test.B, test.U, test.T, mu2, test.Gamma);
        auto K2     = computeMemoryKernel(model2, method, test.tMax, test.errorGoalHigh, test.block);
        auto KI2    = computeCurrentKernel(model2, rI, method, test.tMax, test.errorGoalHigh, test.block);

        Real I1    = KI1.stationaryCurrent(K1.stationaryState());
        Real I2    = KI2.stationaryCurrent(K2.stationaryState());
        Real dIdmu = (I1 - I2) / dmu;

        EXPECT_LE(std::abs(dIdmu - KC.conductance()), test.errorGoalLow)
            << "I1=" << I1 << " I2=" << I2 << " dIdmu=" << dIdmu << " KC.conductance()=" << KC.conductance();
    }
}