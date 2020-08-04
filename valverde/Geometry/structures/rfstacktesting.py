import numpy as np
import matplotlib.pyplot as plt
import os

import warp as wp

from conductors import RF_stack

wp.setup()

# --Useful constants
mm = wp.mm
kV = wp.kV

# --Create Mesh
# xmesh
wp.w3d.xmmin = -4 * mm
wp.w3d.xmmax = 4 * mm
wp.w3d.nx = 100

# ymesh
wp.w3d.ymmin = -4 * mm
wp.w3d.ymmax = 4 * mm
wp.w3d.ny = 100

# zmesh
wp.w3d.zmmin = 0 * mm
wp.w3d.zmmax = 15 * mm
wp.w3d.nz = 300


# --Setup solver
wp.w3d.solvergeom = wp.w3d.XYZgeom
solver = wp.MRBlock3D()
solver.mgtol = 1e-4

# --Boundary conditions
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
# wp.w3d.boundxy = wp.neumann
wp.w3d.boundxy = wp.periodic

# --Register
wp.registersolver(solver)

positions = [np.array([1 * mm, 4 * mm, 9 * mm, 12 * mm])]
voltages = [np.array([3 * kV, -3 * kV])]
stacks = RF_stack(positions, voltages)

wp.package("w3d")
wp.generate()
wp.installconductor(stacks)
wp.fieldsol(-1)

x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh
# --Find zero indices for y
if 0 in wp.w3d.ymesh:
    zeroy = np.where(wp.w3d.ymesh == 0)[0][0]
else:
    # If no 0 in ymesh, take one grid cell off-axis
    zeroy = np.where(wp.w3d.ymesh > 0)[0][0]


# --Warp plots
# wp.winon()
# wp.pfzx(plotselfe=1, comp='z')
# wp.fma()
# wp.pfzx(plotselfe=1, comp='x')
# wp.fma()
# stacks.drawzx()
# wp.fma()


# --Grab fields
phi = wp.getphi()
Ex = wp.getselfe(comp="x")
Ey = wp.getselfe(comp="y")
Ez = wp.getselfe(comp="z")

# --Plot Potential fields
# Potential in zx
Z, X = np.meshgrid(z, x)
phizx = phi[:, zeroy, :]

# Plot and labels
fig, ax = plt.subplots()
ax.set_xlabel("z [mm]")
ax.set_ylabel("x [mm]")
ax.set_title(r"$\Phi(x, y=0, z)$ in zx")

# Create contours
cont = ax.contour(Z / mm, X / mm, phizx, levels=50)
contcb = fig.colorbar(cont, extend="both", shrink=0.8, label=r"$\Phi$ [V]")

# Set zero contour to red -- line
# Check if zero contour exists
if 0 in cont.levels:
    zerocontour = np.where(cont.levels == 0)[0][0]
    zerocontour = cont.collections[zerocontour]
    # set color, linestyle, and width
    zerocontour.set_color("r")
    zerocontour.set_linestyle("dashed")
    zerocontour.set_linewidth(0.5)
else:
    print("No zero contour")

plt.tight_layout()
plt.savefig(os.getcwd() + "/phizx.png")
plt.show()

# --Plot Electric fields
# get Ez and Ex in zx
Ezzx = Ez[:, zeroy, :]
Exzx = Ex[:, zeroy, :]

# Plot and labels Ez first
fig, ax = plt.subplots()
ax.set_xlabel("z [mm]")
ax.set_ylabel("x [mm]")
ax.set_title(r"$E_z(x, y=0, z)$ in zx")

# Create contours
cont = ax.contour(Z / mm, X / mm, Ezzx, levels=50)
contcb = fig.colorbar(cont, extend="both", shrink=0.8, label=r"$E_z$ [V/m]")

# Set zero contour to red -- line
# Check if zero contour exists
if 0 in cont.levels:
    zerocontour = np.where(cont.levels == 0)[0][0]
    zerocontour = cont.collections[zerocontour]
    # set color, linestyle, and width
    zerocontour.set_color("r")
    zerocontour.set_linestyle("dashed")
    zerocontour.set_linewidth(0.5)
else:
    print("No zero contour")

plt.tight_layout()
plt.savefig(os.getcwd() + "/Ezzx.png")
plt.show()

# Plot and labels for Ex
fig, ax = plt.subplots()
ax.set_xlabel("z [mm]")
ax.set_ylabel("x [mm]")
ax.set_title(r"$E_x(x, y=0, z)$ in zx")

# Create contours
cont = ax.contour(Z / mm, X / mm, Exzx, levels=50)
contcb = fig.colorbar(cont, extend="both", shrink=0.8, label=r"$E_x$ [V/m]")

# Set zero contour to red -- line
# Check if zero contour exists
if 0 in cont.levels:
    zerocontour = np.where(cont.levels == 0)[0][0]
    zerocontour = cont.collections[zerocontour]
    # set color, linestyle, and width
    zerocontour.set_color("r")
    zerocontour.set_linestyle("dashed")
    zerocontour.set_linewidth(0.5)
else:
    print("No zero contour")

plt.tight_layout()
plt.savefig(os.getcwd() + "/Exzx.png")
plt.show()
