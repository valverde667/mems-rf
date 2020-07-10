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

wp.w3d.zmmax = 12*mm
wp.w3d.zmmin = 0
wp.w3d.nz = 250


wp.w3d.solvergeom = wp.w3d.XYZgeom

wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.neumann

solver = wp.MRBlock3D()
solver.mgtol = 1e-4
wp.registersolver(solver)

class Ring(object):
    def __init__(self, rmin=1*mm, rmax=1.5*mm, length=2*mm,
                 zcent=0
                 ):
        self.rmin = rmin
        self.rmax = rmax
        self.length = length
        self.zcent = zcent

    def createconductor(self, voltage):
        conductor = wp.ZAnnulus(rmin=rinner, rmax=router, length=length,
                          zcent=zc1, voltage=voltage)

        return conductor

V = 10*kV

rinner = 1*mm
router = 1.5*mm
length = 1*mm
zc1 = 3*mm
zc2 = 8*mm

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
#
# plate1.drawzy(color='fg', filled=True)
# wp.limits(wp.w3d.xmmin, wp.w3d.xmmax, wp.w3d.ymmin, wp.w3d.ymmax)
# plate2.drawzx(color='fg', filled=True)
# wp.pfzx()
# wp.pfzx(scale=1, plotselfe=1, comp='x')
# wp.fma()

#--Plot conductors in xy using zero contours
# Find z-index where first conductor lives
x = wp.w3d.xmesh
y = wp.w3d.ymesh
z = wp.w3d.zmesh


zindexrange = np.where((z>2.5*mm) & (z<3.5*mm))
zindex = zindexrange[0][int(len(zindexrange)/2)] #take center point of range
potential = wp.getphi()[1:-1, 1:-1, zindex]

fig, ax = plt.subplots()
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_title(r'$\Phi(x,y)$')

X,Y = np.meshgrid(y[1:-1], x[1:-1])
cont = ax.contourf(X/mm, Y/mm, potential, levels=50)

if (V in cont.levels ):
    index = np.where(cont.levels == V)[0][0]
else:
    print("V not in levels")

cont.collections[-1].set_linestyle('dashed')
cont.collections[-1].set_color('r')
plt.tight_layout()
plt.show()



Ex = wp.getselfe(comp='x')[1:-1, 1:-1, zindex]
Ey = wp.getselfe(comp='y')[1:-1, 1:-1, zindex]

E = Ex + Ey

X,Y = np.meshgrid(y[1:-1], x[1:-1])

fig, ax = plt.subplots()
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_title(r'$E_x + E_y$')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

contx = ax.contourf(X/mm, Y/mm, E, levels=50)
#--Find 0 contour
if (0 in contx.levels):
    zeroindex = np.where(contx.levels==0)[0][0]
else:
    print("Zero contour doesn't exist")

contx.collections[zeroindex].set_linestyle('dashed')
contx.collections[zeroindex].set_color('r')


plt.tight_layout()
plt.show()


#--Find index where y=0
#Check if it exists
if (0 in wp.w3d.ymesh):
    yindex = np.where(wp.w3d.ymesh==0)[0][0]
#if not find closest positive grid point
else:
    yindex = np.where(wp.w3d.ymesh>0)[0][0]

phizx = wp.getphi()[1:-1, yindex, 1:-1]
Ex = wp.getselfe(comp='x')[1:-1, yindex, 1:-1]
Ez = wp.getselfe(comp='z')[1:-1, yindex, 1:-1]
z,x = np.meshgrid(wp.w3d.zmesh[1:-1], wp.w3d.xmesh[1:-1]) #ignore endpoints

fig,ax = plt.subplots()
phiax = ax.contour(z/mm, x/mm, phizx, levels=40)
#--Find 0 contour if exists (it should)
zeroindex = np.where(phiax.levels==0)[0][0]
phiaxcp = plt.colorbar(phiax, extend='both')

phiax.collections[zeroindex].set_linestyle('dashed')
phiax.collections[zeroindex].set_color('r')
ax.set_xlim(-1,13)
ax.set_ylim(-5, 5)

ax.set_title(r'$\Phi(x, y=0, z)$ vs z')
ax.set_xlabel('z [mm]')
ax.set_ylabel('x [mm]')
phiaxcp.set_label(r'$\Phi(x, y=0, z)$')

plt.tight_layout()
plt.savefig('phi.png')
plt.show()


fig,ax = plt.subplots()
Exax = ax.contour(z/mm, x/mm, Ex, levels=40)
excb = plt.colorbar(Exax, extend='both')
zeroindex = np.where(Exax.levels==0)[0][0]

excb.set_label(r'$E_x(x, y=0, z)$')
Exax.collections[zeroindex].set_linestyle('dashed')
Exax.collections[zeroindex].set_color('r')

ax.set_xlim(-1,13)
ax.set_ylim(-5, 5)
ax.set_title(r'$E_x(x, y=0, z)$ vs z')
ax.set_xlabel('z [mm]')
ax.set_ylabel('x [mm]')

plt.tight_layout()
plt.savefig('Ex.png')
plt.show()
