"""Example script for creating time dependent conductors."""

import numpy as np
import matplotlib.pyplot as plt
import warp as wp
import warp.utils.timedependentvoltage as tdvolt
import pdb

# Set up 3D simulation mesh
wp.w3d.xmmin = -1.5 * wp.mm
wp.w3d.xmmax = 1.5 * wp.mm

wp.w3d.ymmin = -1.5 * wp.mm
wp.w3d.ymmax = 1.5 * wp.mm

wp.w3d.zmmin = -2 * wp.mm
wp.w3d.zmmax = 2 * wp.mm

wp.w3d.nx, wp.w3d.ny = 20, 20
wp.w3d.nz = 200

wp.top.dt = 1e-9

# Specify solver geometry and boundary conditions
wp.w3d.solvergeom = wp.w3d.XYZgeom
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.neumann

# Specify and register solver
solver = wp.MultiGrid3D()
wp.registersolver(solver)
solver.mgtol = 1
solver.mgparam = 1.5
solver.downpasses = 2
solver.uppasses = 2
wp.package("w3d")
wp.generate()  # Create mesh
# wp.ldosolve = False  # Only solve for fields from conductors not particles.
x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh  # Set variable names for ease

# Voltage paramters
MHz = 1e6
Vmax = 10 * wp.kV
Emax = Vmax / 1 / wp.mm
frequency = 14.86 * MHz

# I will now create the capacitor. I'm going to use an annulus with inner
# diameter 0.8mm and outer diameter 1.1mm (.3mm of conducting material) and
# have the two annuluses be symmetric about the origin. I will then create
# the conductor to have a time varying voltage of the form Vsin(ft).
# I must first create a function that takes 'time' as the input and returns a
# voltage. Then I can use this to specify the time-dependence in the conductor
# defintion.
def getvolt(time):
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

    voltage = Vmax * np.cos(frequency * time)
    return voltage


def inv_getvolt(time):
    """Calculate voltage at current time and invert it.

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

    voltage = Vmax * np.cos(frequency * time)
    return -1 * voltage


# Create left and right conductors
left = wp.ZAnnulus(
    rmin=0.8 * wp.mm, rmax=1.1 * wp.mm, length=0.2 * wp.mm, zcent=-1 * wp.mm
)
right = wp.ZAnnulus(
    rmin=0.8 * wp.mm, rmax=1.1 * wp.mm, length=0.2 * wp.mm, zcent=1 * wp.mm
)

# Install conductors on mesh
wp.installconductors(left)
wp.installconductors(right)

# Set conductors to be time dependent
tdvolt.TimeVoltage(left, voltfunc=getvolt, doitnow=True)
tdvolt.TimeVoltage(right, voltfunc=inv_getvolt, doitnow=True)

# Calculate time for one period and set simulation time to stop then.
period = 1 / frequency
wp.top.tstop = period / 2
# Create cgm setup for potential contours
wp.setup()
wp.winon(winnum=1, suffix="pfzx", xon=0)
wp.winon(winnum=2, suffix="pot", xon=0)
while wp.top.time < wp.top.tstop:
    print(getvolt(wp.top.time) / wp.kV)
    # Plot potential contours and conductors
    wp.window(1)
    wp.pfzx(
        fill=1,
        filled=1,
        plotselfe=1,
        plotphi=0,
        comp="z",
        cmin=-Emax,
        cmax=Emax,
        contours=40,
        titlet="RF Electric Field in z-x",
    )
    wp.limits(z.min(), z.max(), x.min(), x.max())
    wp.fma()

    wp.window(2)
    potential = wp.getphi()[int(wp.w3d.nx / 2), int(wp.w3d.ny / 2), :]
    wp.plg(potential, z)
    wp.ptitles("Potential at r = 0 ", "z", "Potential [V]")
    wp.limits(z.min(), z.max(), -Vmax, Vmax)
    wp.fma()

    wp.step()
