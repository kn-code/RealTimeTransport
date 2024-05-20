// clang-format off
#include <RealTimeTransport/Models/DoubleDot.h>
#include <RealTimeTransport/RenormalizedPT/CurrentKernel.h>
#include <RealTimeTransport/RenormalizedPT/MemoryKernel.h>

using namespace SciCore;
using namespace RealTimeTransport;

int main()
{
    // Double dot parameters
    Real Vg    = -1;  // Gate voltage
    Real U     = 5;   // Coulomb interaction
    Real Omega = 2;   // Hybridisation
    Real V     = 0.5; // Bias voltage
    RealVector T{{1, 1}}; // Temperatures leads
    RealVector mu{{V/2, -V/2}}; // Chem. potential

    // Serial setup: lead L - dot 1 - dot 2 - lead R
    RealVector Gamma1{{1, 0}}; // Dot 1 - lead L
    RealVector Gamma2{{0, 1}}; // Dot 2 - lead R

    // Create model
    auto model = createModel<DoubleDot>(
        Vg, Vg, U, Omega, T, mu, Gamma1, Gamma2);

    // Computation parameters
    Real tMax    = 10;    // Maximum simulation time
    Real errGoal = 1e-6;  // Interpolation error goal

    // Memory kernel computation
    auto K = computeMemoryKernel(model,
        RenormalizedPT::Order::_1, tMax, errGoal);

    // Compute stationary state and print it
    Matrix rho = K.stationaryState();
    std::cout << "rho =\n" << rho << "\n";

    // Current kernel computation
    int r    = 0; // Left lead
    auto K_I = computeCurrentKernel(model, r,
        RenormalizedPT::Order::_1, tMax, errGoal);

    // Compute current and print it
    Real I = K_I.stationaryCurrent(rho);
    std::cout << "I = " << I << "\n";

    return 0;
}
