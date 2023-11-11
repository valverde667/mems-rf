"""Test zgridcrossing data dump"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb

import warp as wp

# Set up 3D mesh
wp.w3d.xmmin = -5 * wp.cm
wp.w3d.xmmax = 5 * wp.cm
wp.w3d.nx = 10

wp.w3d.ymmin = -5 * wp.cm
wp.w3d.ymmax = 5 * wp.cm
wp.w3d.ny = 10

wp.w3d.zmmin = -10 * wp.cm
wp.w3d.zmmax = 10 * wp.cm
wp.w3d.nz = 200

# Set boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.dirichlet

# Set 2D solver
solver = wp.MultiGrid3D()
wp.registersolver(solver)

# Initialize particle beam
# beam = wp.Species(type=wp.Potassium, charge_state=+1, name="Beam species")
# beam.ekin = 80.*wp.kV
# wp.top.npmax = 20

wp.package("w3d")
wp.generate()
