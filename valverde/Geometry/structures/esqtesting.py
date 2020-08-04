import numpy as np
import matplotlib.pyplot as plt
import os

import warp as wp

from conductors import ESQ

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
wp.w3d.zmmax = 2 * mm
wp.w3d.nz = 200


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

# --Create ESQ at mesh center
# Initialize inputs for ESQ
center = 1 * mm
voltage = 2 * kV
invertPolarity = True
# Create ESQ
esq = ESQ(center, invertPolarity, voltage)

wp.package("w3d")
wp.generate()

# --When using MRBlock3D the conductors should be installed after the generate
#  function and then the fields recalculated using wp.fieldsol(-1)
wp.installconductor(esq)
wp.fieldsol(-1)

# Simplify mesh variables
x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh

# --Find zero indices. This is useful for plotting especially when plotting in xy.
# zero x-index
if 0 in x:
    zerox = np.where(x == 0)[0][0]
else:
    zerox = np.where(x > 0)[0][0]
# zero y-index
if 0 in y:
    zeroy = np.where(y == 0)[0][0]
else:
    zeroy = np.where(y > 0)[0][0]

# Find z-index where ESQ center is
if center in z:
    zcenterindex = np.where(z == center)[0][0]
else:
    zcenterindex = np.where(z > center)[0][0]

# --Some warp plotting for comparison
# wp.winon(0)
# esq.drawzx()
# wp.fma()
#
# esq.drawxy(iz=zcenterindex)
# wp.fma()
# wp.winon()
# wp.limits(wp.w3d.zmmin/mm, wp.w3d.zmmax/mm, wp.w3d.xmmin/mm, wp.w3d.xmmax/mm)
# esq.drawzx(color='fg', filled=True)
# wp.fma()
#

# --Create mesh for contour plotting
X, Y = np.meshgrid(x, y)
# Get potential at the center of esq in xy
phi = wp.getphi()[:, :, zcenterindex]

# --Plot potential in xy at z=center of esq
fig, ax = plt.subplots()
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_title("Potential of ESQ in x-y")

cont = ax.contour(X / mm, Y / mm, phi, levels=50)
contcb = fig.colorbar(cont, extend="both", shrink=0.8)

# Set zero contour to red -- line
zerocontour = np.where(cont.levels == 0)[0][0]
zerocontour = cont.collections[zerocontour]
zerocontour.set_color("r")
zerocontour.set_linestyle("dashed")
zerocontour.set_linewidth(0.5)

plt.tight_layout()
plt.savefig(os.getcwd() + "/phixy.png")
plt.show()


# --Plot z-x Potential
Z, X = np.meshgrid(z, x)
# Get potential Phi(x, y=0, z)
phi = wp.getphi()[:, zeroy, :]

fig, ax = plt.subplots()
ax.set_xlabel("z [mm]")
ax.set_ylabel("x [mm]")
ax.set_title("Potential of ESQ in z-x")

phi = wp.getphi()[:, zeroy, :]
cont = ax.contour(Z / mm, X / mm, phi, levels=50)
contcb = fig.colorbar(cont, extend="both", shrink=0.8)

# Set zero contour to red -- line
zerocontour = np.where(cont.levels == 0)[0][0]
zerocontour = cont.collections[zerocontour]
zerocontour.set_color("r")
zerocontour.set_linestyle("dashed")
zerocontour.set_linewidth(0.5)

plt.tight_layout()
plt.savefig(os.getcwd() + "/phizx.png")
plt.show()

# --Plot Electric fields
Ex = wp.getselfe(comp="x")
Ey = wp.getselfe(comp="y")
Ez = wp.getselfe(comp="z")

# --Plot Ex field in x-y and z-x
# Get xy for the x-component of the electric field at z-center of esq
Exxy = Ex[:, :, zcenterindex]
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots()
# x-y plot
ax.set_title(r"$E_x$ in x-y")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
xycont = ax.contour(X / mm, Y / mm, Exxy, levels=50)
xycontcb = fig.colorbar(xycont, extend="both", shrink=0.8)

# Set zero contour to red -- line
zerocontour = np.where(xycont.levels == 0)[0][0]
zerocontour = xycont.collections[zerocontour]
zerocontour.set_color("r")
zerocontour.set_linestyle("dashed")
zerocontour.set_linewidth(0.5)
plt.tight_layout()
plt.savefig(os.getcwd() + "/Exxy.png")
plt.show()

# z-x plot
# Get zx for the x-component electric field Ex(x, y=0, z)
Exzx = Ex[:, zeroy, :]
Z, X = np.meshgrid(z, x)

fig, ax = plt.subplots()
ax.set_title(r"$E_x$ in z-x")
ax.set_xlabel("z [mm]")
ax.set_ylabel("x [mm]")
zxcont = ax.contour(Z / mm, X / mm, Exzx, levels=50)
zxcontcb = fig.colorbar(zxcont, extend="both", shrink=0.8)

# Set zero contour to red -- line
zerocontour = np.where(zxcont.levels == 0)[0][0]
zerocontour = zxcont.collections[zerocontour]
zerocontour.set_color("r")
zerocontour.set_linestyle("dashed")
zerocontour.set_linewidth(0.5)
plt.tight_layout()
plt.savefig(os.getcwd() + "/Exzx.png")
plt.show()

# --Plot Ez
# Get zx for z-component of the electric field Ez(x, y=0, z)
Ezzx = Ez[:, zeroy, :]
Z, X = np.meshgrid(z, x)

fig, ax = plt.subplots()
ax.set_title(r"$E_z$ in z-x")
ax.set_xlabel("z [mm]")
ax.set_ylabel("x [mm]")
zxcont = ax.contour(Z / mm, X / mm, Ezzx, levels=50)
zxcontcb = fig.colorbar(zxcont, extend="both", shrink=0.8)

# Set zero contour to red -- line
zerocontour = np.where(zxcont.levels == 0)[0][0]
zerocontour = zxcont.collections[zerocontour]
zerocontour.set_color("r")
zerocontour.set_linestyle("dashed")
zerocontour.set_linewidth(0.5)
plt.tight_layout()
plt.savefig(os.getcwd() + "/Ezzx.png")
plt.show()
