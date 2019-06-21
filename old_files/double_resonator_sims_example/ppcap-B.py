"""
RF model for quarter wavelength system
"""
from warp import *
import numpy as np
import geometry
from geometry import Aperture, ESQ, RF_stack, Gap
import matplotlib.pyplot as plt

#w3d.solvergeom = w3d.XYZgeom
w3d.solvergeom = w3d.XZgeom # 2D
top.pline1 = "1/4 Wavelength Simulation"

# Parameters available for scans
top.dt = 5e-11

Vmax = 1000

setup(prefix="Cap-{}-kV".format(int(Vmax)))

derivqty()

# --- Set input parameters describing the 3d simulation
w3d.l4symtry = True
w3d.l2symtry = False

# ---   for field solve
w3d.bound0 = dirichlet
w3d.boundnz = dirichlet
w3d.boundxy = neumann

# ---   for particles
top.pbound0 = absorb
top.pboundnz = absorb
top.prwall = .004

# --- Set field grid size
w3d.xmmin = -0.005
w3d.xmmax = +0.005
w3d.ymmin = -0.005
w3d.ymmax = +0.005
w3d.zmmin = 0.0
w3d.zmmax = 0.015

# set grid spacing
w3d.nx = 50*4. #scale wrt beam radius
w3d.ny = 50*4. #scale wrt beam radius
w3d.nz = 100.

if w3d.l4symtry:
    w3d.xmmin = 0.
    w3d.nx /= 2
if w3d.l2symtry or w3d.l4symtry:
    w3d.ymmin = 0.
    w3d.ny /= 2

solver = MultiGrid2DDielectric()
registersolver(solver)

# --- Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.)
package("w3d")
generate()

Vmax = 1000

ID_RF = 201
geometry.pos = 0
rfgap = 1.5e-3 - 2*5e-6

#A = ZCylinder(4*mm, 3*mm, zcent=0, voltage=Vmax)
#B = ZCylinder(4*mm, 3*mm, zcent=.015, voltage=0)

geometry.Gap(3*mm)
RF = geometry.RF_stack(condid=[ID_RF,ID_RF+1,ID_RF+2,ID_RF+3], rfgap = rfgap, drift = 0.7e-3, voltage=Vmax)
geometry.Gap(8*mm)

installconductors(RF)
#solver.epsilon[:,2s0:60]*=10000
#solver.epsilon[0:50,:]*=10
#solver.epsilon[40:50,30:60]*=1000
solver.epsilon[20:50,20:30]*=1.
solver.epsilon[20:50,35:44]*=1.

fieldsol(-1)

winon(xon=0)

pfzx(fill=1, filled=1, plotselfe=2, comp='E', titles=0,cmin=0, cmax=1e6)

ptitles("Electric field", "Z [m]", "X [m]", "")

fma()
