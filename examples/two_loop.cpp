// clang-format off

#include <fstream>

#include <SciCore/Parallel.h>

#include <RealTimeTransport/Models/AndersonDot.h>
#include <RealTimeTransport/IteratedRG/CurrentKernel.h>
#include <RealTimeTransport/IteratedRG/MemoryKernel.h>

#include "helper.h"

using namespace RealTimeTransport;
using namespace SciCore;

void plot(const std::string& jsonDataFile, const std::string& pythonPlotFile);

int main()
{
    // Create Anderson dot with two reservoirs
    Real epsilon = -4;        // Dot energy
    Real B       =  2;        // Magnetic field
    Real U       = 10;        // Coulomb repulsion
    RealVector T{{0, 0}};     // Temperature leads
    RealVector mu{{1, -1}};   // Chem. potential
    RealVector Gamma{{1, 1}}; // Tunnel rate to leads
    auto model = createModel<AndersonDot>(
        epsilon, B, U, T, mu, Gamma);

    // Computation parameters
    Real tMax    = 20;   // Maximum simulation time
    Real errGoal = 1e-4; // Interpolation error goal
    int r = 0;
    int block = 0;

    // Set up model and define parameter as before
    // Compute memory kernel & propagator
    tf::Executor executor(8);
    auto method = IteratedRG::Order::_2;
    auto K  = computeMemoryKernel(model,
        method, tMax, errGoal, executor);
    auto Pi = computePropagator(K);

    // Compute one block of the current kernel
    auto K_I = computeCurrentKernel(K, Pi, r,
        method, tMax, errGoal, executor, block);

    // Set initial state: Basis 0, Up, Down, UpDown
    Matrix rho0 = Matrix::Zero(4, 4);
    rho0.diagonal() << 0, 1, 0, 0;

    // Compute occupations Tr{nUp*rho(t)}
    RealVector t = RealVector::LinSpaced(1000, 0, tMax);

    // Compute current kernel & transient current
    auto I   = computeCurrent(K_I, Pi, rho0);
    RealVector I_t = I(t);

    // Save data as .json file
    std::string jsonDataFile = "data-two-loop.json";
    {
        std::ofstream os(jsonDataFile);
        cereal::JSONOutputArchive archive(os);
        archive(CEREAL_NVP(epsilon), CEREAL_NVP(B), CEREAL_NVP(U), CEREAL_NVP(T), CEREAL_NVP(mu), CEREAL_NVP(Gamma),
                CEREAL_NVP(tMax), CEREAL_NVP(errGoal), CEREAL_NVP(t), CEREAL_NVP(I_t));
    }

    // Generate python script for plotting
    std::string pythonPlotFile = "plot-two-loop.py";
    plot(jsonDataFile, pythonPlotFile);
    std::cout << "I_RG = " << K_I.stationaryCurrent(K.stationaryState()) << "\n";
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

t = np.array(data['t'])
I = np.array(data['I_t'])

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 12
})

plt.plot(t, I, label=r'$I_{RG}(t)$')
plt.legend()
plt.xlabel(r'$t \Gamma$')
plt.savefig('out.png', dpi=600, bbox_inches='tight')
plt.show()
)";

    replaceTag(pythonScript, "{jsonDataFile}", jsonDataFile);

    std::ofstream out(pythonPlotFile);
    out << pythonScript;
    out.close();
}