import numpy as np
import matplotlib.pyplot as plt

import warp as wp

kV = wp.kV
mm = wp.mm


wp.w3d.xmmin = -2 * mm
wp.w3d.xmmax = 2 * mm
wp.w3d.nx = 100

wp.w3d.ymmin = -1 * mm
wp.w3d.ymmax = 1 * mm
wp.w3d.ny = 100

wp.w3d.zmmin = 0
wp.w3d.zmmax = 10 * mm
wp.w3d.nz = 200

solver = wp.MRBlock3D()
wp.registersolver(solver)


def esq(voltage, radius=0.5 * mm, center=1 * mm, zc=3 * mm, length=2 * mm):
    # --Create pole in quadrant 1
    pole1 = wp.ZCylinder(
        radius=radius,
        length=length,
        voltage=voltage,
        xcent=center,
        ycent=center,
        zcent=zc,
    )

    # --Create pole in quadrant 2
    pole2 = wp.ZCylinder(
        radius=radius,
        length=length,
        voltage=-voltage,
        xcent=-center,
        ycent=center,
        zcent=zc,
    )

    # --Create pole in quadrant 3
    pole3 = wp.ZCylinder(
        radius=radius,
        length=length,
        voltage=voltage,
        xcent=-center,
        ycent=-center,
        zcent=zc,
    )

    # --Create pole in quadrant 4
    pole4 = wp.ZCylinder(
        radius=radius,
        length=length,
        voltage=-voltage,
        xcent=center,
        ycent=-center,
        zcent=zc,
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
phixy = wp.getphi()[:, :, zcenterindex]

fig, ax = plt.subplots()
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
X, Y = np.meshgrid(x, y)
cont = ax.contour(X / mm, Y / mm, phixy, levels=50)
cb = fig.colorbar(cont)
plt.show()
