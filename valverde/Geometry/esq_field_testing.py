import numpy as np
import matplotlib.pyplot as plt

import warp as wp

kV = wp.kV
mm = wp.mm


wp.w3d.xmmin = -2 * mm
wp.w3d.xmmax = 2 * mm
wp.w3d.nx = 100

wp.w3d.ymmin = -2 * mm
wp.w3d.ymmax = 2 * mm
wp.w3d.ny = 100

wp.w3d.zmmin = 0
wp.w3d.zmmax = 10 * mm
wp.w3d.nz = 200

solver = wp.MRBlock3D()
wp.registersolver(solver)


def esq(voltage, radius=0.5 * mm, center=1 * mm, zc=3 * mm, length=2 * mm):
    # --Create top pole
    pole1 = wp.ZCylinder(
        radius=radius, length=length, voltage=voltage, xcent=0, ycent=center, zcent=zc,
    )

    # --Create left pole
    pole2 = wp.ZCylinder(
        radius=radius,
        length=length,
        voltage=-voltage,
        xcent=-center,
        ycent=0,
        zcent=zc,
    )

    # --Create bottom pole
    pole3 = wp.ZCylinder(
        radius=radius, length=length, voltage=voltage, xcent=0, ycent=-center, zcent=zc,
    )

    # --Create right pole
    pole4 = wp.ZCylinder(
        radius=radius, length=length, voltage=-voltage, xcent=center, ycent=0, zcent=zc,
    )

    # --Add elements for total quadrupole

    quad = pole1 + pole2 + pole3 + pole4

    return quad


voltage = 1 * kV
quad = esq(voltage)
wp.installconductors(quad)
wp.generate()

x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh

xzeroindex = np.transpose(np.where(x == 0))[0, 0]
yzeroindex = np.transpose(np.where(y == 0))[0, 0]
zcenterindex = np.transpose(np.where(z == 3 * mm))[0, 0]

wp.setup()
quad.drawzx(filled=True)
wp.limits(0, 10 * mm, -2 * mm, 2 * mm)
wp.fma()
quad.drawxy(filled=True)
wp.fma()
wp.limits(-2 * mm, 2 * mm, -2 * mm, 2 * mm)
wp.pfxy(iz=yzeroindex,)
wp.fma()
wp.pfzx(fill=1, filled=1)
wp.fma()

raise Exception()


phixy = wp.getphi()[:, :, zcenterindex]

fig, ax = plt.subplots()
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
X, Y = np.meshgrid(x, y)
cont = ax.contour(X / mm, Y / mm, phixy, levels=50)
cb = fig.colorbar(cont)
plt.show()
