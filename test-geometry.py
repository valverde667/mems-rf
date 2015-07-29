from warp import *
import numpy as np

from geometry import RF_stack
from helper import gitversion

# which geometry to use 2d or 3d
# w3d.solvergeom = w3d.RZgeom
w3d.solvergeom = w3d.XYZgeom

# define some strings that go into the output file
top.pline1 = "Test geometries"
top.pline2 = " " + gitversion()
top.runmaker = "Arun Persaud (apersaud@lbl.gov)"

# --- Invoke setup routine for the plotting
setup()

# ---   for field solve
w3d.bound0 = dirichlet
w3d.boundnz = neumann
w3d.boundxy = neumann

# ---   for particles
top.pbound0 = absorb
top.pboundnz = absorb
top.prwall = np.sqrt(2)*1.5*mm/2.0

# --- Set field grid size
w3d.xmmin = -0.0015/2.
w3d.xmmax = +0.0015/2.
w3d.ymmin = -0.0015/2.
w3d.ymmax = +0.0015/2.
w3d.zmmin = 0.0
w3d.zmmax = 0.003

if w3d.l4symtry:
    w3d.xmmin = 0.
if w3d.l2symtry or w3d.l4symtry:
    w3d.ymmin = 0.

# set grid spacing
w3d.dx = (w3d.xmmax-w3d.xmmin)/100.
w3d.dy = (w3d.ymmax-w3d.ymmin)/100.
w3d.dz = (w3d.zmmax-w3d.zmmin)/100.

# --- Field grid dimensions - note that nx and ny must be even.
w3d.nx = 2*int((w3d.xmmax - w3d.xmmin)/w3d.dx/2.)
w3d.xmmax = w3d.xmmin + w3d.nx*w3d.dx
w3d.ny = 2*int((w3d.ymmax - w3d.ymmin)/w3d.dy/2.)
w3d.ymmax = w3d.ymmin + w3d.ny*w3d.dy
w3d.nz = int((w3d.zmmax - w3d.zmmin)/w3d.dz)
w3d.zmmax = w3d.zmmin + w3d.nz*w3d.dz


# --- Set up fieldsolver - 7 means the multigrid solver
top.fstype = 7
f3d.mgtol = 1.0  # Poisson solver tolerance, in volts
f3d.mgparam = 1.5
f3d.downpasses = 2
f3d.uppasses = 2

# --- Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.)
package("w3d")
generate()

conductors = RF_stack(voltage=1e3, zcenter=3*mm/2., condid=[201, 202, 203, 204])

# define the electrodes
installconductors(conductors)

# --- Recalculate the fields
fieldsol(-1)

winon()

for i in range(100):
    fma()
    pfxy(fill=0, filled=1, plotphi=0, iz=i)

for i in range(100):
    fma()
    pfzx(fill=0, filled=1, plotphi=0, iy=i)

