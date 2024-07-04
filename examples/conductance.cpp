// clang-format off
#include <fstream>
#include <iostream>

#include <SciCore/Parallel.h>

#include <RealTimeTransport/Models/AndersonDot.h>
#include <RealTimeTransport/RenormalizedPT/ConductanceKernel.h>

#include "helper.h"

using namespace RealTimeTransport;
using namespace SciCore;

void plot(const std::string& jsonDataFile, const std::string& pythonPlotFile);

int main()
{
    // Anderson dot parameters
    Real B = 0;
    Real U = 10;
    RealVector T{{1, 1}};
    RealVector mu{{0, 0}};
    RealVector Gamma{{1, 1}};

    // Simulation parameters
    auto method  = RenormalizedPT::Order::_2;
    Real tMax    = 5;
    Real errGoal = 1e-3;
    int block    = 0;

    // Swipe ε and compute the conductance dI/dV at each point
    RealVector epsilon = RealVector::LinSpaced(61, -2 * U, U);
    RealVector dIdV(epsilon.size());
    tf::Executor executor;
    parallelFor(
        [&](int i)
        {
            auto model = createModel<AndersonDot>(epsilon[i], B, U, T, mu, Gamma);
            auto K     = computeMemoryKernel(model, method, tMax, errGoal, block);
            auto KI    = computeCurrentKernel(model, block, method, tMax, errGoal, block);
            auto KC    = computeConductanceKernel(K, KI, method, block);

            dIdV[i] = KC.conductance();
        }, 0, epsilon.size(), executor);

    //
    // Plotting
    //

    // Save data as .json file
    std::string jsonDataFile = "data-conductance.json";
    {
        std::ofstream os(jsonDataFile);
        cereal::JSONOutputArchive archive(os);
        archive(
            CEREAL_NVP(B), CEREAL_NVP(U), CEREAL_NVP(T), CEREAL_NVP(mu), CEREAL_NVP(Gamma), CEREAL_NVP(tMax),
            CEREAL_NVP(errGoal), CEREAL_NVP(epsilon), CEREAL_NVP(dIdV));
    }

    // Generate python script for plotting
    std::string pythonPlotFile = "plot-conductance.py";
    plot(jsonDataFile, pythonPlotFile);
    std::cout << "Created python file for plotting: " << pythonPlotFile << "\n";

    return 0;
}

void plot(const std::string& jsonDataFile, const std::string& pythonPlotFile)
{
    std::string pythonScript =

        R"(
import matplotlib.pyplot as plt
import numpy as np
import json

with open('{jsonDataFile}', 'r') as f:
    data = json.load(f)

U = data['U']
epsilon = np.array(data['epsilon'])
dIdV = np.array(data['dIdV'])
G0 = 1/np.pi

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 12
})

plt.plot(epsilon/U, dIdV/G0, label=r'$G/G_0$')
plt.legend()
plt.xlabel(r'$\epsilon/U$')
plt.savefig('out.png', dpi=600, bbox_inches='tight')
plt.show()
)";

    replaceTag(pythonScript, "{jsonDataFile}", jsonDataFile);

    std::ofstream out(pythonPlotFile);
    out << pythonScript;
    out.close();
}
