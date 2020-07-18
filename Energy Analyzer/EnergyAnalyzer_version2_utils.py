import numpy as np
import os, json, datetime, time, math, importlib, sys, scipy
from scipy.constants import elementary_charge
import matplotlib.pyplot as plt
from warp import *

pathtoparticlefiles = (
    "/home/timo/Documents/LBL/Warp/atap-meqalac-simulations/Energy Analyzer/v2_beams/"
)


def savejson(data, name):
    with open(name, "w") as writefile:
        json.dump(data, writefile, sort_keys=True, indent=1)


def readjson(name):
    fp = f"{name}"
    with open(fp, "r") as readfile:
        data = json.load(readfile)
    return data


def savePlost(name="plost"):
    pass


def lostByConductor(condid=30):
    uz = []
    ux = []
    uy = []
    zz = []
    xx = []
    yy = []
    for id, uzp, uxp, uyp, z, x, y in zip(top.pidlost[:, -1], top.uzplost, top.uxplost, top.uyplost, top.zplost,
                                          top.xplost, top.yplost):
        if id == condid:
            uz.append(uzp)
            ux.append(uxp)
            uy.append(uyp)
            zz.append(z)
            xx.append(x)
            yy.append(yy)
    return uz, ux,uy,zz,xx,yy

# def histograms(prefilter,postfilter):
#     fig, (ax1, ax2) = plt.subplots(2)
#     ax1.hist(prefilter["ekinZ"], bins=range(4000,110000,1000))
#     ax2.hist(postfilter["ekinZ"], bins=range(4000,110000,500))
#     plt.savefig("histograms.png")
#     plt.savefig("histograms.svg")
def histograms(prefilter, postfilter):
    fig, (ax1, ax2) = plt.subplots(2)
    # ax1.hist(prefilter["ekinZ"], bins=range(4000,110000,1000))
    ax2.hist(postfilter["ekinZ"],
             range=(2, 10000))  # , bins=range(4000,110000,500))
    plt.savefig("histograms.png")
    plt.savefig("histograms.svg")


def velocity(ekin):
    return np.sqrt(2 * ekin * echarge / (40 * amu))


def energy(v):
    return 1 / 2 * 40 * amu * v * v / echarge


# geometry
n_beams = 4


# Deflector Plates
def deflectionPlates(voltage):
    centerpositionZ = 53 * mm  # b
    distanceplates = 25 * mm  # d
    # Dimensions of metal plates
    plateX = 1 * mm
    plateY = 50 * mm
    plateZ = 50.8 * mm  # c
    # x-shift
    x2 = -6 * mm  # e
    x1 = x2 + distanceplates
    plate1 = Box(
        xsize=plateX,
        ysize=plateY,
        zsize=plateZ,
        xcent=x1,
        ycent=0,
        zcent=centerpositionZ,
        voltage=-voltage / 2,
        condid=20,
    )
    plate2 = Box(
        xsize=plateX,
        ysize=plateY,
        zsize=plateZ,
        xcent=x2,
        ycent=0,
        zcent=centerpositionZ,
        voltage=voltage / 2,
        condid=21,
    )
    return plate1 + plate2


def slit_before_plates():
    # aperture
    # build a box and punch holes in it
    d_aperture = 1 * mm
    # setup
    aperture = Box(
        xsize=40 * mm,
        ysize=40 * mm,
        zsize=0.1 * mm,
        xcent=7 * mm,
        ycent=0,
        zcent=0,
        voltage=0,
        condid=10,
    )
    holes = [
        ZCylinder(
            zcent=0,
            ycent=0,
            xcent=i * 3 * mm,
            radius=d_aperture,
            length=0.1 * mm,
            voltage=0,
        )
        for i in range(n_beams)
    ]
    return aperture - sum(holes)


def filter_plates():
    # filters:
    # build a box and punch holes in it
    n_filters = 3
    phis_filter = [0.07526778307059949, 0.07438315246073729,
                   0.07400417250218974]
    pos_filter_z = [0.1, 0.14, 0.18]  # g
    phi_offsets = [
        (p - pos_filter_z[0]) * np.tan(ph) for p, ph in
        zip(pos_filter_z, phis_filter)
    ]
    thickness_filterplate = 0.5e-3
    filter_apertures = [1e-3, 1e-3, 1e-3]  # diameter
    bias_filters = [10, 20, 30]  # voltage on the plates
    initial_heights = [
        0.003621128630106844,
        0.0065801180436158855,
        0.009557295653103096,
    ]  # f
    #
    print(f"Phi_offset : \n{phi_offsets}")
    #
    filters = []
    for apt, pos, vol, phi_offset in zip(
            filter_apertures, pos_filter_z, bias_filters, phi_offsets
    ):
        plate = Box(
            xsize=100 * mm,
            ysize=50 * mm,
            zsize=thickness_filterplate,
            xcent=initial_heights[0],
            ycent=0,
            zcent=pos,
            voltage=vol,
        )
        holes = [
            ZCylinder(
                zcent=pos,
                ycent=0,
                xcent=initial_height + phi_offset,
                radius=apt / 2,
                length=thickness_filterplate,
                voltage=0,
            )
            for initial_height in initial_heights
        ]
        filters.append(plate - sum(holes))
    return sum(filters)


def tiltedInfiniteBox(z_front, x_front, thickness, angle=0, voltage=0):
    """
        This class can be used to calibrate the setup → find slitpositions
        z and x is a point on the front of the box
        box is infinite in two dimensions
    """
    phi = 0
    theta = angle
    z1 = -tan(theta) * x_front + z_front
    z2 = z1 + thickness / cos(theta)
    print(f"Z1 = {z1}\nZ2 = {z2}  ")
    return Plane(
        z0=z1,
        zsign=1,
        theta=theta,
        phi=phi,
        voltage=voltage,
        condid=30
    ) - Plane(
        z0=z2,
        zsign=1,
        theta=theta,
        phi=phi,
        voltage=voltage,
        condid=30
    )


def tiltedBox(x_cent=0, y_cent=0, z_cent=0, angle=pi / 4, x_dim=1, y_dim=1, z_dim=1, voltage=0):
    # x_dim, y_dim, z_dim are the thicknesses in x, y, z direction
    # __--2-`\
    # \       \
    # 1\   .   \ 3  a
    #   \       \
    #    \__-4-``
    #        c
    assert angle % (pi / 2) != 0
    # P1 minus the others
    # 1
    p1 = Plane(
        zcent=z_cent - z_dim / 2 / cos(angle) + tan(angle) * x_cent,
        zsign=1,
        theta=angle,
        phi=0,
        voltage=voltage,
    )
    p3 = Plane(
        zcent=z_cent + z_dim / 2 / cos(angle) + tan(angle) * x_cent,
        zsign=1,
        theta=angle,
        phi=0,
        voltage=voltage,
    )
    angle2 = angle - pi / 2
    p2 = Plane(
        zcent=z_cent - (x_cent + x_dim / 2 / cos(angle)) / tan(angle),
        zsign=-1,
        theta=angle2,
        phi=0,
        voltage=voltage,
    )
    p4 = Plane(
        zcent=z_cent - (x_cent - x_dim / 2 / cos(angle)) / tan(angle),
        zsign=1,
        theta=angle2,
        phi=0,
        voltage=voltage,
    )
    pYminus = YPlane(ycent=-y_dim / 2 + y_cent, ysign=-1)
    pYplus = YPlane(ycent=+y_dim / 2 + y_cent, ysign=1)
    return p1 - p3 - p2 - p4 - pYminus - pYplus


def slittedTiltedBox(entranceSlits, angle, slitwidth, z_dim, x_dim, y_dim, voltage=0):
    """
    entranceSlits: z|x coordinates of the center a slit
    angle: rotation along y Axis
    slitwidth: width of the slit
    """
    z_overshoot = z_dim * 1.1  # 10% overshoot to account for entrance slit not being exactly on the surface of the tilted box

    def to_center(z, x):
        return z + z_dim / 2 * cos(angle), x + z_dim / 2 * sin(angle)

    z0, x0 = entranceSlits[0]
    z_cent, x_cent = to_center(z0, x0)

    tb = tiltedBox(x_cent=x_cent, z_cent=z_cent, angle=angle, z_dim=z_dim, x_dim=x_dim, y_dim=y_dim, voltage=voltage)

    slits = []
    for es in entranceSlits:
        zz, xx = to_center(es[0], es[1])
        slits.append(tiltedBox(x_cent=xx, z_cent=zz, angle=angle, z_dim=z_overshoot, x_dim=slitwidth, y_dim=y_dim))
        print(f"zz = {zz}\txx = {xx}")

    return tb - sum(slits)


""" 100kev 5500V
Angles : [0.0740217  0.0749499  0.07424881 0.07617565]
Angles deg : [4.24113082 4.29431277 4.25414349 4.36454305]
Energies : [99447.6691773  99441.0940278  99449.41455825 99433.72478108]
z pos : [0.09912104 0.09956524 0.09934374 0.09978553]
x pos : [0.01246544 0.00653563 0.00949254 0.00359491]

"""
