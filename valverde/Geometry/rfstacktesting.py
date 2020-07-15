import numpy as np
import matplotlib.pyplot as plt
import os

import warp as wp

from conductors import RF_stack

wp.setup()

#--Useful constants
mm = wp.mm
kV = wp.kV

#--Create Mesh
#xmesh
wp.w3d.xmmin = -4*mm
wp.w3d.xmmax = 4*mm
wp.w3d.nx = 100

#ymesh
wp.w3d.ymmin = -4*mm
wp.w3d.ymmax = 4*mm
wp.w3d.ny = 100

#zmesh
wp.w3d.zmmin = 0*mm
wp.w3d.zmmax = 15*mm
wp.w3d.nz = 300


#--Setup solver
wp.w3d.solvergeom = wp.w3d.XYZgeom
solver = wp.MRBlock3D()
solver.mgtol = 1e-4

#--Boundary conditions
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
#wp.w3d.boundxy = wp.neumann
wp.w3d.boundxy = wp.periodic

#--Register
wp.registersolver(solver)

positions = [1*mm, 4*mm, 9*mm, 12*mm]
voltages = [-3*kV, 3*kV, -3*kV, 3*kV]
stacks = RF_stack(positions, voltages)

wp.package('w3d')
wp.generate()
wp.installconductors(stacks)
wp.fieldsol(-1)

x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh

wp.winon()
wp.pfzx()
wp.fma()
