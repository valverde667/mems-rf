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


####
