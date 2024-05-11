// clang-format off

#include <RealTimeTransport/Models/AndersonDot.h>
#include <RealTimeTransport/RenormalizedPT/CurrentKernel.h>
#include <RealTimeTransport/RenormalizedPT/MemoryKernel.h>

using namespace RealTimeTransport;
using namespace SciCore;

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
    auto t = RealVector::LinSpaced(100, 0, tMax);
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

    return 0;
}
