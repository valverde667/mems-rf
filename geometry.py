"""Module to keep track of different geometries used in our project"""
from __future__ import print_function

from warp import *


# universial dimensions

# unit cell frame
framelength = 1500*um
framewidth = 150*um

# wafer dimenions
wafer_body = 500*um
wafer_box = 2*um
wafer_si = 20*um
wafer_length = wafer_body + wafer_box + wafer_si

# globals
pos = 0


def Gap(dist=500*um):
    """Vacuum gap, e.g. between wafers"""
    global pos
    pos += dist


def ESQ(voltage, condid):
    """Simple ESQ wafer

    Use 3 cylinders in Z to make up a single electrode.
    Add to current position(pos) and use two condids [a,b] for the electrods.

    Bias these +-+- with voltage V.

    """
    global pos
    R1 = 96*um  # center cylinder
    R2 = 75*um  # outdise cylinders

    X = -337*um  # X offset for outer electrodes
    Y = 125*um  # +-Y offset for outer electrodes
    XX = -187*um  # X offset for inner electrode

    zcenter = pos + 0.5*wafer_length

    def element(voltage, condid, rotation):
        """create a single element, rotated X degrees"""

        if rotation == 0:
            xcent1, ycent1 = X, Y
            xcent2, ycent2 = X, -Y
            xcent3, ycent3 = XX, 0
        elif rotation == 90:
            xcent1, ycent1 = Y, -X
            xcent2, ycent2 = -Y, -X
            xcent3, ycent3 = 0, -XX
        elif rotation == 180:
            xcent1, ycent1 = -X, Y
            xcent2, ycent2 = -X, -Y
            xcent3, ycent3 = -XX, 0
        elif rotation == 270:
            xcent1, ycent1 = Y, X
            xcent2, ycent2 = -Y, X
            xcent3, ycent3 = 0, XX
        else:
            print("wrong rotation value")

        electrode1 = ZCylinder(radius=R2, length=wafer_length, voltage=voltage,
                               xcent=xcent1, ycent=ycent1, zcent=zcenter,
                               condid=condid)
        electrode2 = ZCylinder(radius=R2, length=wafer_length, voltage=voltage,
                               xcent=xcent2, ycent=ycent2, zcent=zcenter,
                               condid=condid)
        electrode3 = ZCylinder(radius=R1, length=wafer_length, voltage=voltage,
                               xcent=xcent3, ycent=ycent3, zcent=zcenter,
                               condid=condid)
        return electrode1 + electrode2 + electrode3

    condidA, condidB = condid

    bodycenter = zcenter - 0.5*wafer_length + 250*um  # assume body of SOI is on the left
    Frame1 = Box(framelength, framelength, 500*um,
                 zcent=bodycenter, voltage=+voltage, condid=condidA)
    Frame2 = Box(framelength-2*framewidth, framelength-2*framewidth, 600*um,
                 zcent=bodycenter, voltage=+voltage, condid=condidA)
    InnerBox1 = Box(framelength/2+X, 2*(Y+R2), 500*um,
                    xcent=-framelength/2.+(framelength/2.+X)/2.,
                    zcent=bodycenter, voltage=+voltage, condid=condidA)
    InnerBox2 = Box(framelength/2+X, 2*(Y+R2), 500*um,
                    xcent=framelength/2.-(framelength/2.+X)/2.,
                    zcent=bodycenter, voltage=+voltage, condid=condidA)
    FrameA = (Frame1-Frame2) + InnerBox1 + InnerBox2

    SOIcenter = zcenter + 0.5*wafer_length - 10*um  # assume body of SOI is on the left
    Frame1 = Box(framelength, framelength, 20*um,
                 zcent=SOIcenter, voltage=-voltage, condid=condidB)
    Frame2 = Box(framelength-2*framewidth, framelength-2*framewidth, 30*um,
                 zcent=SOIcenter, voltage=-voltage, condid=condidB)
    InnerBox1 = Box(2*(Y+R2), framelength/2.+X, 20*um,
                    ycent=-framelength/2.+(framelength/2.+X)/2.,
                    zcent=SOIcenter, voltage=-voltage, condid=condidB)
    InnerBox2 = Box(2*(Y+R2), framelength/2.+X, 20*um,
                    ycent=framelength/2.-(framelength/2.+X)/2.,
                    zcent=SOIcenter, voltage=-voltage, condid=condidB)
    FrameB = (Frame1-Frame2) + InnerBox1 + InnerBox2

    # cylinders
    electrodeA = element(voltage=+voltage, condid=condidA, rotation=0)
    electrodeB = element(voltage=-voltage, condid=condidB, rotation=90)
    electrodeC = element(voltage=+voltage, condid=condidA, rotation=180)
    electrodeD = element(voltage=-voltage, condid=condidB, rotation=270)

    pos += wafer_length

    return electrodeA + electrodeB + electrodeC + electrodeD + FrameA + FrameB


def RF_stack(voltage, condid, rfgap=200*um):
    """two wafers with grounded planes on the outside and an RF-gap in the middle"""
    global pos
    condidA, condidB, condidC, condidD = condid

    r_beam = 90*um

    SOIcenter = pos + 0.5*wafer_si
    Frame = Box(framelength, framelength, 20*um,
                zcent=SOIcenter, voltage=-voltage, condid=condidA)
    Beam = ZCylinder(r_beam, 25*um, zcent=SOIcenter, voltage=voltage, condid=condidA)
    SOI1 = Frame-Beam
    pos += wafer_si + wafer_box

    bodycenter = pos + 0.5*wafer_body
    Frame = Box(framelength, framelength, 500*um,
                zcent=bodycenter, voltage=0, condid=condidB)
    Beam = ZCylinder(r_beam, 510*um, zcent=bodycenter, voltage=0, condid=condidB)
    body1 = Frame-Beam
    pos += wafer_body

    pos += rfgap

    bodycenter = pos + 0.5*wafer_body
    Frame = Box(framelength, framelength, 500*um,
                zcent=bodycenter, voltage=0, condid=condidC)
    Beam = ZCylinder(r_beam, 510*um, zcent=bodycenter, voltage=0, condid=condidC)
    body2 = Frame-Beam
    pos += wafer_body

    pos += wafer_box

    SOIcenter = pos + 0.5*wafer_si
    Frame = Box(framelength, framelength, 20*um,
                zcent=SOIcenter, voltage=voltage, condid=condidD)
    Beam = ZCylinder(r_beam, 25*um, zcent=SOIcenter, voltage=voltage, condid=condidD)
    SOI2 = Frame-Beam
    pos += wafer_si

    return SOI1 + body1 + body2 + SOI2


def RF_stack2(condid, rfgap=200*um, voltage=0):
    """two wafers with a gap of rfgap between them. Both wafers are
    insulating with conducting layers on both sides"""

    global pos
    condidA, condidB, condidC, condidD = condid

    r_beam = 90*um
    thickness = 2*um

    SOIcenter = pos + 0.5*thickness
    Frame = Box(framelength, framelength, thickness, voltage=0,
                zcent=SOIcenter, condid=condidA)
    Beam = ZCylinder(r_beam, 5*um, zcent=SOIcenter, voltage=0, condid=condidA)
    SOI1 = Frame-Beam
    pos += thickness + 500*um

    bodycenter = pos + 0.5*thickness
    Frame = Box(framelength, framelength, thickness, voltage=voltage,
                zcent=bodycenter, condid=condidB)
    Beam = ZCylinder(r_beam, 5*um, zcent=bodycenter, voltage=voltage, condid=condidB)
    body1 = Frame-Beam
    pos += thickness

    pos += rfgap

    bodycenter = pos + 0.5*thickness
    Frame = Box(framelength, framelength, thickness, voltage=voltage,
                zcent=bodycenter, condid=condidC)
    Beam = ZCylinder(r_beam, 5*um, zcent=bodycenter, voltage=voltage, condid=condidC)
    body2 = Frame-Beam
    pos += thickness + 500*um

    SOIcenter = pos + 0.5*thickness
    Frame = Box(framelength, framelength, thickness, voltage=0,
                zcent=SOIcenter, condid=condidD)
    Beam = ZCylinder(r_beam, 5*um, zcent=SOIcenter, voltage=0, condid=condidD)
    SOI2 = Frame-Beam
    pos += thickness

    return SOI1 + body1 + body2 + SOI2


def RF_stack3(condid, rfgap=200*um, gap=500*um, voltage=0):
    """4 wafers rf stack with 2 acceleration stages

     The first two form an RF acceleration cell, followed by a drift
     of length rfgap, followed by a second RF accererlation cell.

    """

    global pos
    condidA, condidB, condidC, condidD = condid

    r_beam = 90*um
    thickness = 2*um + 500*um + 2*um

    SOIcenter = pos + 0.5*thickness
    Frame = Box(framelength, framelength, thickness, voltage=0,
                zcent=SOIcenter, condid=condidA)
    Beam = ZCylinder(r_beam, thickness + 50*um, zcent=SOIcenter, voltage=0, condid=condidA)
    Ground1 = Frame-Beam
    pos += thickness + gap

    bodycenter = pos + 0.5*thickness
    Frame = Box(framelength, framelength, thickness, voltage=voltage,
                zcent=bodycenter, condid=condidB)
    Beam = ZCylinder(r_beam, thickness + 50*um, zcent=bodycenter, voltage=voltage, condid=condidB)
    body1 = Frame-Beam
    pos += thickness

    pos += rfgap

    bodycenter = pos + 0.5*thickness
    Frame = Box(framelength, framelength, thickness, voltage=voltage,
                zcent=bodycenter, condid=condidC)
    Beam = ZCylinder(r_beam, thickness + 50*um, zcent=bodycenter, voltage=voltage, condid=condidC)
    body2 = Frame-Beam
    pos += thickness + 500*um

    SOIcenter = pos + 0.5*thickness
    Frame = Box(framelength, framelength, thickness, voltage=0,
                zcent=SOIcenter, condid=condidD)
    Beam = ZCylinder(r_beam, thickness + 50*um, zcent=SOIcenter, voltage=0, condid=condidD)
    Ground2 = Frame-Beam
    pos += thickness

    return Ground1 + body1 + body2 + Ground2


def Aperture(voltage, condid, width=10*um):
    """A simple thin wafer with a whole for the beam"""

    global pos

    r_beam = 90*um
    bodycenter = pos + 0.5*width

    Frame = Box(framelength, framelength, width,
                zcent=bodycenter, voltage=voltage, condid=condid)
    Beam = ZCylinder(r_beam, 2*width, zcent=bodycenter, voltage=voltage, condid=condid)
    body = Frame-Beam

    pos += width

    return body
