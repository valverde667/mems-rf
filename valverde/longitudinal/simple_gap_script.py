import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy.integrate as integrate
import os
import math
import csv
import pdb
import sys

# Useful constants
kV = 1e3
mm = 1e-3
um = 1e-6

# Create argument parser for scaling. Must be done before importing Warp
import warpoptions

# Scale pole argument will set the radius of the ESQ rod in units of aperture
# radius
warpoptions.parser.add_argument("--center", default=False, type=float)

# Scale length argument will set the length of the ESQ rod in units of aperture
# radius
warpoptions.parser.add_argument("--voltage", default=False, type=float)

inputs = warpoptions.parser.parse_args()
if inputs.center != False:
    center = inputs.center
else:
    # Around optimum value found for isolated single quad.
    center = 0.0 * mm

if inputs.voltage != False:
    voltage = inputs.voltage
else:
    # Actual length of 0.695 * mm
    voltage = 0.5 * kV

import warp as wp


# ------------------------------------------------------------------------------
#                     User Defined function
# Section createst the conductor classes for loading onto the mesh as well as
# some utility functions to be used.
# ------------------------------------------------------------------------------
def getindex(mesh, value, spacing):
    """Find index in mesh for or mesh-value closest to specified value

    Function finds index corresponding closest to 'value' in 'mesh'. The spacing
    parameter should be enough for the range [value-spacing, value+spacing] to
    encompass nearby mesh-entries .

    Parameters
    ----------
    mesh : ndarray
        1D array that will be used to find entry closest to value
    value : float
        This is the number that is searched for in mesh.
    spacing : float
        Dictates the range of values that will fall into the region holding the
        desired value in mesh. Best to overshoot with this parameter and make
        a broad range.

    Returns
    -------
    index : int
        Index for the mesh-value closest to the desired value.
    """

    # Check if value is already in mesh
    if value in mesh:
        return np.where(mesh == value)[0][0]

    # Create array of possible indices
    indices = np.where((mesh > (value - spacing)) & (mesh < (value + spacing)))[0]

    # Compute differences of the indexed mesh-value with desired value
    difference = []
    for index in indices:
        diff = np.sqrt((mesh[index] ** 2 - value ** 2) ** 2)
        difference.append(diff)

    # Smallest element will be the index closest to value in indices
    i = np.argmin(difference)
    index = indices[i]

    return index


# ------------------------------------------------------------------------------
#                     Create and load mesh and conductors
# ------------------------------------------------------------------------------

# Creat mesh using conductor geometries (above) to keep resolution consistent
wp.w3d.xmmin = -5 * mm
wp.w3d.xmmax = 5 * mm
wp.w3d.nx = 100

wp.w3d.ymmin = -5 * mm
wp.w3d.ymmax = 5 * mm
wp.w3d.ny = 100

# Calculate nz to get about designed dz
wp.w3d.zmmin = -10 * mm
wp.w3d.zmmax = 10 * mm
wp.w3d.nz = 200

# Add boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.dirichlet
wp.f3d.mgtol = 1e-7

wp.w3d.l2symtry = False
solver = wp.MRBlock3D()
wp.registersolver(solver)

# Create left and right quads
conductor = wp.Annulus(
    rmin=3.0 * mm,
    rmax=4.0 * mm,
    length=1.0 * mm,
    voltage=voltage,
    zcent=center,
    xcent=0.0 * mm,
    ycent=0.0 * mm,
)
wp.installconductor(conductor)
wp.generate()


# Rename meshes and find indicesfor the mesh z-center and z-center of right quad
x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh
zzeroindex = getindex(z, 0.0, wp.w3d.dz)
zcenterindex = getindex(z, center, wp.w3d.dz)
xzeroindex = getindex(x, 0.0, wp.w3d.dx)
yzeroindex = getindex(y, 0.0, wp.w3d.dy)

# Create Warp plots. Useful for quick-checking
warpplots = True
if warpplots:
    wp.setup()
    conductor.drawzx(filled=True)
    # rightwall.drawzx(filled=True)
    # leftwall.drawzx(filled=True)
    wp.fma()

    conductor.drawxy(filled=True)
    wp.fma()

    wp.pfxy(iz=zcenterindex, fill=1, filled=1)
    wp.fma()

    wp.pfzx(fill=1, filled=1)
    wp.fma()

    wp.pfxy(
        plotselfe=1,
        plotphi=0,
        comp="x",
        fill=1,
        filled=1,
        contours=100,
        iz=zcenterindex,
    )
    wp.fma()

    wp.pfxy(
        plotselfe=1,
        plotphi=0,
        comp="y",
        fill=1,
        filled=1,
        contours=100,
        iz=zcenterindex,
    )
    wp.fma()

# Grab Fields
phi = wp.getphi()
phixy = wp.getphi()[:, :, zcenterindex]
Ex = wp.getselfe(comp="x")
Ey = wp.getselfe(comp="y")
