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
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.neumann

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

x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh
#--Find zero indices
#zero x-index
if (0 in x):
    zerox = np.where(x==0)[0][0]
else:
    zerox = np.where(x>0)[0][0]
#zero y-index
if (0 in y):
    zeroy = np.where(y==0)[0][0]
else:
    zeroy = np.where(y>0)[0][0]

#--Find z-index where ESQ center is
if (center in z):
    zcenterindex = np.where(z==center)[0][0]
else:
    zcenterindex = np.where(z>center)[0][0]


# wp.winon()
# wp.pfxy(iz=zcenterindex)
# wp.fma()
# wp.winon()
# wp.limits(wp.w3d.zmmin/mm, wp.w3d.zmmax/mm, wp.w3d.xmmin/mm, wp.w3d.xmmax/mm)
# esq.drawzx(color='fg', filled=True)
# wp.fma()
#
X,Y = np.meshgrid(x,y)
phi = wp.getphi()[:, :, zcenterindex]

#--Plot potential in xy at z=center of esq
fig, ax = plt.subplots()
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_title('Potential of ESQ in x-y')
cont = ax.contour(X/mm, Y/mm, phi, levels=50 )
contcb = fig.colorbar(cont, extend='both', shrink=0.8)
#Set zero contour to red -- line
zerocontour = np.where(cont.levels==0)[0][0]
zerocontour = cont.collections[zerocontour]
zerocontour.set_color('r')
zerocontour.set_linestyle('dashed')
zerocontour.set_linewidth(0.5)

plt.show()


#--Plot z-x Potential
Z,X = np.meshgrid(z,x)
phi = wp.getphi()[:,zeroy,:]

fig, ax = plt.subplots()
ax.set_xlabel('z [mm]')
ax.set_ylabel('x [mm]')
ax.set_title('Potential of ESQ in z-x')

phi = wp.getphi()[:,zeroy,:]
cont = ax.contour(Z/mm, X/mm, phi, levels=50 )
contcb = fig.colorbar(cont, extend='both', shrink=0.8)

#Set zero contour to red -- line
zerocontour = np.where(cont.levels==0)[0][0]
zerocontour = cont.collections[zerocontour]
zerocontour.set_color('r')
zerocontour.set_linestyle('dashed')
zerocontour.set_linewidth(0.5)

plt.show()
