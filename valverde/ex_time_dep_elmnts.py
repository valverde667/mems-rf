"""Example script for creating time dependent conductors."""

import numpy as np
import matplotlib.pyplot as plt
import warp as wp
import pdb

# Set up 3D simulation mesh
wp.w3d.xmmin = -1.5 * wp.mm
wp.w3d.xmmax = 1.5 * wp.mm

wp.w3d.ymmin = -1.5 * wp.mm
wp.w3d.ymmax = 1.5 * wp.mm

wp.w3d.zmmin = -2 * wp.mm
wp.w3d.zmmax = 2 * wp.mm

wp.w3d.nx, wp.w3d.ny = 100, 100
wp.w3d.nz = 200

wp.top.dt = 1e-10
wp.top.tstop = 200 * wp.top.dt

# Specify solver geometry and boundary conditions
wp.w3d.solvergeom = wp.w3d.XYZgeom
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.neumann

# Specify and register solver
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

# Voltage paramters
MHz = 1e6
Vmax = 10 * wp.kV
frequency = 14.86 * MHz

# I will now create the capacitor. I'm going to use an annulus with inner
# diameter 0.8mm and outer diameter 1.1mm (.3mm of conducting material) and
# have the two annuluses be symmetric about the origin. I will then create
# the conductor to have a time varying voltage of the form Vsin(ft).
# I must first create a function that takes 'time' as the input and returns a
# voltage. Then I can use this to specify the time-dependence in the conductor
# defintion.
def get_voltage(time):
    """Calculate voltage at current time.

    Function calculates the voltage at the current timestep in the simulation.
    The global variables are needed for proper calculations and cannot be
    specified in the function call or Warp will reject the function.

    Parameters
    ----------
    time : float
        Current simulation time.

    Returns
    -------
    voltage : float
        Current voltage using sinusoidal varying voltage.

    """

    global Vmax, frequency

    voltage = Vmax * np.sin(frequency * time)
    return voltage


####
