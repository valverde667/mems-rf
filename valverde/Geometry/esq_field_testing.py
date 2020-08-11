import numpy as np
import matplotlib.pyplot as plt

import warp as wp

kV = wp.kV
mm = wp.mm
um = 1e-6

wp.w3d.xmmin = -4 * mm
wp.w3d.xmmax = 4 * mm
wp.w3d.nx = 150

wp.w3d.ymmin = -4 * mm
wp.w3d.ymmax = 4 * mm
wp.w3d.ny = 150

wp.w3d.zmmin = -5 * mm
wp.w3d.zmmax = 5 * mm
wp.w3d.nz = 400

wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.neumann

solver = wp.MRBlock3D()
wp.registersolver(solver)


class ESQ:
    def __init__(self, radius=0.75 * mm, zc=2.2 * mm, length=1.59 * mm):

        self.radius = radius
        self.zc = zc
        self.length = length

    def pole(self, voltage, xcent, ycent):
        conductor = wp.ZCylinder(
            voltage=voltage,
            xcent=xcent,
            ycent=ycent,
            zcent=self.zc,
            radius=self.radius,
            length=self.length,
        )
        return conductor

    def generate(self, voltage, xcent, ycent, data=False):
        top = self.pole(voltage=voltage, xcent=0, ycent=ycent)
        bottom = self.pole(voltage=voltage, xcent=0, ycent=-ycent)
        left = self.pole(voltage=-voltage, xcent=-xcent, ycent=0)
        right = self.pole(voltage=-voltage, xcent=xcent, ycent=0)

        components = [top, bottom, left, right]
        conductor = top + bottom + left + right

        if data == False:
            return conductor
        else:
            return conductor, components


class Wall:
    def __init__(self, rextent=100 * w3d.xmmax, zextent=0.1 * mm):
        self.rextent = rextent
        self.zextent = zextent

    def generate(self, apperture, voltage, zcenter):
        conductor = ZAnnulus(
            rmin=apperture,
            voltage=voltage,
            zcent=zcenter,
            rmax=self.rextent,
            length=self.zextent,
        )

        return conductor


# def esq(voltage, radius=0.5 * mm, center=1 * mm, zc=3 * mm, length=625 * um):
#     # --Create top pole
#     pole1 = wp.ZCylinder(
#         radius=radius, length=length, voltage=voltage, xcent=0, ycent=center, zcent=zc,
#     )
#
#     # --Create left pole
#     pole2 = wp.ZCylinder(
#         radius=radius,
#         length=length,
#         voltage=-voltage,
#         xcent=-center,
#         ycent=0,
#         zcent=zc,
#     )
#
#     # --Create bottom pole
#     pole3 = wp.ZCylinder(
#         radius=radius, length=length, voltage=voltage, xcent=0, ycent=-center, zcent=zc,
#     )
#
#     # --Create right pole
#     pole4 = wp.ZCylinder(
#         radius=radius, length=length, voltage=-voltage, xcent=center, ycent=0, zcent=zc,
#     )
#
#     # --Add elements for total quadrupole
#
#     quad = pole1 + pole2 + pole3 + pole4
#
#     return quad


voltage = 0.5 * kV
xycent = 1.3 * mm
separation = 1 * mm
zc = 2.2 * mm
length = 1.59 * mm


leftconductor = ESQ(zc=-zc)
leftquad, leftcomponents = leftconductor.generate(
    voltage=voltage, xcent=xycent, ycent=xycent, data=True
)
rightconductor = ESQ(zc=zc)
rightquad, rightcomponents = rightconductor.generate(
    voltage=-voltage, xcent=xycent, ycent=xycent, data=True
)


# quad = esq(voltage)

wp.installconductor(leftquad)
wp.installconductor(rightquad)
wp.generate()

x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh

xzeroindex = np.transpose(np.where(x == 0))[0, 0]
yzeroindex = np.transpose(np.where(y == 0))[0, 0]
zcenterindex = np.transpose(np.where(z == 3 * mm))[0, 0]

# wp.setup()
# quadclass1.drawzx(filled=True)
# wp.limits(0, 10 * mm, -2 * mm, 2 * mm)
# wp.fma()
# quadclass1.drawxy(filled=True)
# wp.fma()
# wp.limits(-2 * mm, 2 * mm, -2 * mm, 2 * mm)
# wp.pfxy(iz=yzeroindex,)
# wp.fma()
# wp.pfzx(fill=1, filled=1)
# wp.fma()

# wp.setup()
# wp.limits(0, 10 * mm, -2 * mm, 2 * mm)
# quadclass2.drawzx(filled=True)
# wp.fma()
# quadclass2.drawxy(filled=True)
# wp.fma()
# wp.pfxy(iz=yzeroindex,)
# wp.fma()
# wp.pfzx(fill=1, filled=1)
# wp.fma()
#
# wp.setup()
# rightquad.drawzx(filled=True)
# leftquad.drawzx(filled=True)
# wp.fma()
# leftquad.drawxy(filled=True)
# rightquad.drawxy(filled=True)
# wp.fma()
# wp.pfxy(iz=yzeroindex)
# wp.fma()
# wp.pfzx()
# wp.fma()

phi = wp.getphi()
phixy = wp.getphi()[:, :, zcenterindex]
Ex = wp.getselfe(comp="x")
gradex = Ex[xzeroindex + 1, yzeroindex, :] / wp.w3d.dx

fig, ax = plt.subplots()
ax.set_xlabel("z [mm]")
ax.set_ylabel(r"$E_x(dx, 0, z)$/dx [V mm$^{-2}$]")
ax.set_title(r"$E_x$ Gradient One Grid-cell Off-axis vs z")
ax.scatter(z / mm, gradex / mm / mm, s=1.2)
ax.axhline(y=0, c="k", lw=0.5)
ax.axvline(x=0, c="k", lw=0.5)

# add ESQ
esq1left = -zc - length / 2
esq1right = -zc + length / 2

esq2left = zc - length / 2
esq2right = zc + length / 2
ax.axvline(x=esq1left / mm, c="b", lw=0.8, ls="--", label="First ESQ")
ax.axvline(x=esq1right / mm, c="b", lw=0.8, ls="--")
ax.axvline(x=esq2left / mm, c="r", lw=0.8, ls="--", label="Second ESQ")
ax.axvline(x=esq2right / mm, c="r", lw=0.8, ls="--")
plt.legend()
plt.show()

# fig, ax = plt.subplots()
# ax.set_xlabel("x [mm]")
# ax.set_ylabel("y [mm]")
# X, Y = np.meshgrid(x, y)
# cont = ax.contour(X / mm, Y / mm, phixy, levels=50)
# cb = fig.colorbar(cont)
# plt.show()
