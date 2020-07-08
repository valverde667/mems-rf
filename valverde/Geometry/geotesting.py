import numpy as np
import matplotlib.pyplot as plt

import warp as wp

from conductors import ESQ

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
wp.w3d.zmmax = 2*mm
wp.w3d.nz = 200


#--Setup solver
wp.w3d.solvergeom = wp.w3d.XYZgeom
solver = wp.MRBlock3D()
solver.mgtol = 1e-4

#--Boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.dirichlet

#--Register
wp.registersolver(solver)

#--Create ESQ at mesh center
center = 1*mm
voltage = 2*kV
invertPolarity = True
esq = ESQ(center, invertPolarity, voltage)

wp.package('w3d')
wp.generate()

wp.installconductor(esq)
wp.fieldsol(-1)

#--Find zero indices
#zero x-index
if (0 in wp.w3d.xmesh):
    zerox = np.where(wp.w3d.xmesh==0)[0][0]
else:
    zerox = np.where(wp.w3d.xmesh>0)[0][0]
#zero y-index
if (0 in wp.w3d.xmesh):
    zeroy = np.where(wp.w3d.ymesh==0)[0][0]
else:
    zeroy = np.where(wp.w3d.ymesh>0)[0][0]

#--Find z-index where ESQ center is
if (center in wp.w3d.zmesh):
    zcenterindex = np.where(wp.w3d.zmesh==center)[0][0]
else:
    zcenterindex = np.where(wp.w3d.zmesh>center)[0][0]
