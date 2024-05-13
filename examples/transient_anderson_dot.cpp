// clang-format off

#include <fstream>

#include <SciCore/Serialization.h>

#include <RealTimeTransport/Models/AndersonDot.h>
#include <RealTimeTransport/RenormalizedPT/CurrentKernel.h>
#include <RealTimeTransport/RenormalizedPT/MemoryKernel.h>

#include "helper.h"

using namespace RealTimeTransport;
using namespace SciCore;

void plot(const std::string& jsonDataFile, const std::string& pythonPlotFile);

int main()
{
    // Create Anderson dot with two reservoirs
    Real epsilon = -4;        // Dot energy
    Real B       = -1;        // Magnetic field
    Real U       = 10;        // Coulomb repulsion
    RealVector T{{0, 0}};     // Temperature leads
    RealVector mu{{2, -2}};   // Chem. potential
    RealVector Gamma{{1, 1}}; // Tunnel rate to leads
    auto model = createModel<AndersonDot>(
        epsilon, B, U, T, mu, Gamma);

    // Computation parameters
    Real tMax    = 5;    // Maximum simulation time
    Real errGoal = 1e-4; // Interpolation error goal
    int block    = 0;    // Only compute first block

    // Compute memory kernel & propagator
    auto method = RenormalizedPT::Order::_2;
    auto K  = computeMemoryKernel(model,
        method, tMax, errGoal, block);
    auto Pi = computePropagator(K, block);

    // Set initial state: Basis 0, Up, Down, UpDown
    Matrix rho0 = Matrix::Zero(4, 4);
    rho0.diagonal() << 0, 1, 0, 0;

    // Operator nUp
    Matrix nUp = Matrix::Zero(4, 4);
    nUp.diagonal() << 0, 1, 0, 1;

    // Compute occupations Tr{nUp*rho(t)}
    RealVector t = RealVector::LinSpaced(100, 0, tMax);
    RealVector n(t.size());
    for (int i = 0; i < t.size(); ++i)
        n[i] = (Pi(t[i], rho0) * nUp).trace().real();

    // Print
    std::cout << "t = " << t.transpose() << "\n"
              << "n = " << n.transpose() << "\n";

    // Compute current kernel & transient current
    int r    = 0; // Left reservoir
    auto K_I = computeCurrentKernel(model, r, method, tMax, errGoal, block);
    auto I   = computeCurrent(K_I, Pi, rho0);
    RealVector I_t = I(t);

    // Print
    std::cout << "I = " << I_t.transpose() << "\n";

    // Save data as .json file
    std::string jsonDataFile = "data-transient-anderson-dot.json";
    {
        std::ofstream os(jsonDataFile);
        cereal::JSONOutputArchive archive(os);
        archive(CEREAL_NVP(epsilon), CEREAL_NVP(B), CEREAL_NVP(U), CEREAL_NVP(T), CEREAL_NVP(mu), CEREAL_NVP(Gamma),
                CEREAL_NVP(tMax), CEREAL_NVP(errGoal), CEREAL_NVP(t), CEREAL_NVP(n), CEREAL_NVP(I_t));
    }

    // Generate python script for plotting
    std::string pythonPlotFile = "plot-transient-anderson-dot.py";
    plot(jsonDataFile, pythonPlotFile);
    std::cout << "Created python file for plotting: " << pythonPlotFile << "\n";
    return 0;

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
n = np.array(data['n'])
I = np.array(data['I_t'])

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 12
})

plt.plot(t, n, label=r'$n(t)$')
plt.plot(t, I, label=r'$I(t)$')
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