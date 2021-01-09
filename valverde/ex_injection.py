"""Script for testing Warp settings with injection"""

import numpy as np
import matplotlib.pyplot as plt
import warp as wp
import pdb

# Set up a 3D simulation mesh
wp.w3d.xmmin = -1.5 * wp.mm
wp.w3d.xmmax = 1.5 * wp.mm

wp.w3d.ymmin = -1.5 * wp.mm
wp.w3d.ymmax = 1.5 * wp.mm

wp.w3d.zmmin = 0
wp.w3d.zmmax = 23 * wp.mm

wp.w3d.nx, wp.w3d.ny = 70, 70
wp.w3d.nz = 180

wp.top.dt = 1e-10
wp.top.tstop = 200 * wp.top.dt

# Set species of particle
ions = wp.Species(type=wp.Argon, charge_state=1, name="Ar")
new_ions = wp.Species(type=wp.Nitrogen, charge_state=1, name="N")
# Set up boundary conditions  for simulation and specify geometry
wp.w3d.solvergeom = wp.w3d.XYZgeom
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.neumann
wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb

# Set up particle input quantities
wp.top.ns = 2  # Number of species
# wp.top.npmax = 10
# wp.top.np_s = [10, 10]  # Number of particles by species
wp.top.ibeam_s = [10e-6, 10e-6]  # Current magnitude of particles by species
# wp.top.ekin_s = [7 * wp.kV]  # Kinetic energy of particles by species
wp.top.vbeam_s = [1e6, 1.8e6 * np.random.rand()]
wp.derivqty()  # Evaluate global constants used in Warp

wp.top.lsavelostpart = False  # Save lost particles

# Set up history settings
lspeciesmoments = True  # Save mmnts for both species and combined mmnts (last entry)
wp.top.nhist = 1  # Set history saving to every timestep. Defaults at 5.
wp.top.lhvzofz = True  # Turn on history of vz

# Set up injection/beam parameters
wp.top.inject = 1  # Constant current injection
wp.top.ainject = 1 * wp.mm  # Width of injection in x
wp.top.binject = 1 * wp.mm  # Width of injection in y
wp.top.apinject = 0 * wp.mm  # Convergence angle of injection in x
wp.top.bpinject = 0 * wp.mm  # Convergence angle of injection in y
wp.top.npinje_s = [1, 2]  # Number of particles injected per species
wp.top.zinject = wp.w3d.zmmin  # z-location of injection
wp.top.vinject = 0  # Injector voltage

# Initialize field solver
solver = wp.MRBlock3D()
wp.registersolver(solver)
solver.ldosolve = False  # Turn off spacecharge
solver.mgtol = 1
solver.mgparam = 1.5
solver.downpasses = 2
solver.uppasses = 2
wp.package("w3d")
wp.generate()  # Create mesh
x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh  # Set variable names for ease
wp.step()  # Step 1 to inject first particle

# Initialize plotting. Setup two cgm files for xy and zx particle plots.
wp.setup()
wp.winon(winnum=1, suffix="xy", xon=0)
wp.winon(winnum=2, suffix="zx", xon=0)

# Make a circle to show the beam pipe
R = 0.5 * wp.mm  # Beam radius
t = np.linspace(0, 2 * np.pi, 100)
X = R * np.sin(t)
Y = R * np.cos(t)

# Main loop. Inject particles and create cgm plots
# pdb.set_trace()
while wp.top.time < wp.top.tstop:
    # Create particle plot in xy
    wp.window(1)
    ions.ppxy(color=wp.red, msize=10)
    new_ions.ppxy(color=wp.blue, msize=10)
    wp.limits(x.min(), x.max(), y.min(), y.max())

    # Add beam pipe to plot as dashed line
    wp.plg(Y, X, type="dash")
    wp.fma()

    # Create particle plot in xz
    wp.window(2)
    ions.ppzx(color=wp.red, msize=10)
    new_ions.ppzx(color=wp.blue, msize=10)
    wp.limits(z.min(), z.max(), x.min(), x.max())
    wp.fma()

    wp.step()
