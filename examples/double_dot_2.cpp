// clang-format off
#include <fstream>
#include <sstream>

#include <SciCore/Parallel.h>

#include <RealTimeTransport/Models/DoubleDot.h>
#include <RealTimeTransport/RenormalizedPT/CurrentKernel.h>
#include <RealTimeTransport/RenormalizedPT/MemoryKernel.h>

using namespace RealTimeTransport;
using namespace SciCore;

#include "helper.h"

void plot(const std::string& jsonDataFile, const std::string& pythonPlotFile);

int main()
{
    // Double dot parameters
    Real U = 5;             // Coulomb interaction
    Real V = 0.5;           // Bias voltage
    RealVector T{{0, 0}};   // Temperature leads

    // We chose the geometry left lead - dot 1 - dot 2 - right lead
    RealVector Gamma1{{1, 0}};  // Couple dot 1 to left lead
    RealVector Gamma2{{0, 1}};  // Couple dot 2 to right lead

    // Simulation parameters
    Real tMax    = 10;
    Real errGoal = 1e-4;

    // Define lambda function that computes the current given Vg and Omega
    auto computeI = [&](Real Vg, Real Omega) -> Real
    {
        // Create model
        auto model = createModel<DoubleDot>(Vg, Vg, U, Omega, T, RealVector{{V / 2, -V / 2}}, Gamma1, Gamma2);

        // Compute memory kernel
        int block = 0;
        RenormalizedPT::MemoryKernel K;
        try
        {
            K = computeMemoryKernel(model, RenormalizedPT::Order::_1, tMax, errGoal, block);
        }
        catch (AccuracyError<RenormalizedPT::MemoryKernel>& error)
        {
            std::cerr << "Accuracy warning at Vg=" << Vg << ", Omega=" << Omega << ": " << error.what() << "\n";
            K = std::move(error.value());
        }

        // Compute current kernel
        int r = 0; // Left lead
        RenormalizedPT::CurrentKernel K_I;
        try
        {
            K_I = computeCurrentKernel(model, r, RenormalizedPT::Order::_1, tMax, errGoal, block);
        }
        catch (AccuracyError<RenormalizedPT::CurrentKernel>& error)
        {
            std::cerr << "Accuracy warning at Vg=" << Vg << ", Omega=" << Omega << ": " << error.what() << "\n";
            K_I = std::move(error.value());
        }

        return K_I.stationaryCurrent(K.stationaryState());
    };

    int numPoints    = 101;
    RealVector Vg    = RealVector::LinSpaced(numPoints, -12.5, 7.5);
    RealVector Omega = RealVector::LinSpaced(numPoints, 0, 5);

    // Solve with 4 threads in parallel
    tf::Executor executor(4);
    RealMatrix I(numPoints, numPoints);
    parallelFor(
        [&](int index)
        {
            int i   = index / numPoints;
            int j   = index % numPoints;
            I(i, j) = computeI(Vg[i], Omega[j]);
        }, 0, numPoints * numPoints, executor);

    // Save data as .json file
    std::stringstream os; os << "data-double-dot-T-" << T[0] << ".json";
    std::string jsonDataFile = os.str();
    {
        std::ofstream os(jsonDataFile);
        cereal::JSONOutputArchive archive(os);
        archive(
            CEREAL_NVP(U), CEREAL_NVP(V), CEREAL_NVP(T), CEREAL_NVP(Gamma1), CEREAL_NVP(Gamma2),
            CEREAL_NVP(tMax), CEREAL_NVP(errGoal), CEREAL_NVP(Vg), CEREAL_NVP(Omega), CEREAL_NVP(I));
    }

    // Generate python script for plotting
    os.str(""); os << "plot-double-dot-T-" << T[0] << ".py";
    std::string pythonPlotFile = os.str();
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

U  = np.array(data['U'])
Vg = np.array(data['Vg'])
Omega = np.array(data['Omega'])
I = np.array(data['I'])

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 12
})

X, Y = np.meshgrid(Vg/U, Omega)
plt.pcolormesh(X, Y, I.T, cmap='RdYlBu')
plt.colorbar(label=r'$I/\Gamma$')
plt.xlabel(r'$V_g/U$')
plt.ylabel(r'$\Omega/\Gamma$')
plt.savefig('out.png', dpi=600, bbox_inches='tight')
plt.show()
)";

    replaceTag(pythonScript, "{jsonDataFile}", jsonDataFile);

    std::ofstream out(pythonPlotFile);
    out << pythonScript;
    out.close();
}