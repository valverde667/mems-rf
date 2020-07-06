import numpy as np
import matplotlib.pyplot as plt

import warp as wp

wp.setup()
mm = wp.mm
kV = wp.kV

#--Create Cube Mesh
wp.w3d.xmmax = 4*mm
wp.w3d.xmmin = -4*mm
wp.w3d.nx = 110

wp.w3d.ymmax = 4*mm
wp.w3d.ymmin = -4*mm
wp.w3d.ny = 100

wp.w3d.zmmax = 6*mm
wp.w3d.zmmin = 0
wp.w3d.nz = 250


wp.w3d.solvergeom = wp.w3d.XYZgeom

wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.dirichlet

solver = wp.MRBlock3D()
solver.mgtol = 1e-4
wp.registersolver(solver)

class Plate(object):
    def __init__(self, xcentr=0, ycentr=0, zcentr=0,
                 length=1, width=1, height=1
                 ):
        self.xcentr = xcentr
        self.ycentr = ycentr
        self.zcentr = zcentr
        self.length = length
        self.width = width
        self.height = height

    def getconductor(self, Voltage):
        conductor = wp.Box(xsize=self.height, ysize=self.width, zsize=self.length,
                           xcent=self.xcentr, ycent=self.ycentr,zcent=self.zcentr,
                           voltage=Voltage)

        return conductor

V = 10*kV

rinner = 1*mm
router = 1.5*mm
length = 1*mm
zc1 = 2*mm
zc2 = 4*mm

plate1 = wp.ZAnnulus(rmin=rinner, rmax=router, length=length,
                  zcent=zc1, voltage=V)
plate2 = wp.ZAnnulus(rmin=rinner, rmax=router, length=length,
                  zcent=zc2, voltage= -V)



wp.package('w3d')
wp.generate()

wp.installconductor(plate1)
wp.installconductor(plate2)
wp.fieldsol(-1)

# wp.winon()
# plate1.draw(color='fg', filled=True)
# plate2.drawzx(color='fg', filled=True)
# wp.pfzx()
# wp.pfzx(scale=1, plotselfe=1, comp='x')
# wp.limits(wp.w3d.zmmin, wp.w3d.zmmax)
# wp.fma()

phizx = wp.getphi()[1:, 1, :]
Ex = wp.getselfe(comp='x')[1:, 1, :]
Ez = wp.getselfe(comp='z')[1:, 1, :]
z,x = np.meshgrid(wp.w3d.zmesh, wp.w3d.xmesh[1:])

fig,ax = plt.subplots()
#phiax = ax.contour(z, x, phizx)
Exax = ax.contour(z, x, Ex, levels=20)
excb = plt.colorbar(Exax, extend='both')
ax.set_xlabel('z')
ax.set_ylabel('x')
plt.tight_layout()
plt.show()
