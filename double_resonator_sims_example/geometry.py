"""Module to keep track of different geometries used in our project"""
from warp import *

# universial dimensions

# unit cell frame
framelength = 5e-3
framewidth = .150e-3

# wafer dimenions
# SOI
ESQ_wafer_body = 500*um
ESQ_wafer_box = 2*um
ESQ_wafer_si = 20*um
ESQ_wafer_length = ESQ_wafer_body + ESQ_wafer_box + ESQ_wafer_si

# RF
#RF_thickness = 2*um + 500*um + 2*um
RF_gap = 200*um

#RF
RF_thickness = 5*um

rf_gap = 1.5e-3 - 2*RF_thickness

r_beam = 1e-3

# globals
pos = 0


def Gap(dist=500*um):
    """Vacuum gap, e.g. between wafers"""
    global pos
    pos += dist
    print("--- Gap ends at: ", pos)


def RF_stack(condid , rfgap=1.5e-3 , drift=0.7e-3 , voltage=0):
    global pos
    condidA, condidB, condidC, condidD = condid
    r_aperture = 1*mm
#grounded plate 1
    wafercenter = pos + 0.5*RF_thickness
    Frame = Box(framelength, framelength, RF_thickness, voltage=0,
                zcent=wafercenter, condid=condidA)
    Beam = ZCylinder(r_aperture, RF_thickness + 5*um, zcent=wafercenter, voltage=0, condid=condidA)
    Ground1 = Frame-Beam
    pos += RF_thickness + rfgap

#RF plate 1
    wafercenter = pos + 0.5*RF_thickness
    Frame = Box(framelength, framelength, RF_thickness, voltage=voltage,
                zcent=wafercenter, condid=condidB)
    Beam = ZCylinder(r_aperture, RF_thickness + 5*um, zcent=wafercenter, voltage=voltage, condid=condidB)
    RF1 = Frame-Beam
    pos += RF_thickness + drift


#RF plate 2
    wafercenter = pos + 0.5*RF_thickness
    Frame = Box(framelength, framelength, RF_thickness, voltage=voltage,
                zcent=wafercenter, condid=condidC)
    Beam = ZCylinder(r_aperture, RF_thickness + 5*um, zcent=wafercenter, voltage=voltage, condid=condidC)
    RF2 = Frame-Beam
    pos += RF_thickness + rfgap

#ground plate 2
    wafercenter = pos + 0.5*RF_thickness
    Frame = Box(framelength, framelength, RF_thickness, voltage=0,
                zcent=wafercenter, condid=condidD)
    Beam = ZCylinder(r_aperture, RF_thickness + 5*um, zcent=wafercenter, voltage=0, condid=condidD)
    Ground2 = Frame-Beam
    pos += RF_thickness

    return Ground1 + RF1 + RF2 + Ground2


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

    print("--- ESQ starts at: ", pos)
    print("--- ESQ voltage: ", voltage)
    zcenter = pos + 0.5*ESQ_wafer_length

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

        electrode1 = ZCylinder(radius=R2, length=ESQ_wafer_length, voltage=voltage,
                               xcent=xcent1, ycent=ycent1, zcent=zcenter,
                               condid=condid)
        electrode2 = ZCylinder(radius=R2, length=ESQ_wafer_length, voltage=voltage,
                               xcent=xcent2, ycent=ycent2, zcent=zcenter,
                               condid=condid)
        electrode3 = ZCylinder(radius=R1, length=ESQ_wafer_length, voltage=voltage,
                               xcent=xcent3, ycent=ycent3, zcent=zcenter,
                               condid=condid)
        return electrode1 + electrode2 + electrode3

    condidA, condidB = condid

    # cylinders
    if callable(voltage):
        pos_voltage = voltage

        def neg_voltage(x):
            return -voltage(x)
    else:
        pos_voltage = voltage
        neg_voltage = -voltage

    bodycenter = zcenter - 0.5*ESQ_wafer_length + ESQ_wafer_body/2  # assume body of SOI is on the left
    Frame1 = Box(framelength, framelength, ESQ_wafer_body,
                 zcent=bodycenter, voltage=pos_voltage, condid=condidA)
    Frame2 = Box(framelength-2*framewidth, framelength-2*framewidth, ESQ_wafer_body*1.1,
                 zcent=bodycenter, voltage=pos_voltage, condid=condidA)
    InnerBox1 = Box(framelength/2+X, 2*(Y+R2), ESQ_wafer_body,
                    xcent=-framelength/2.+(framelength/2.+X)/2.,
                    zcent=bodycenter, voltage=pos_voltage, condid=condidA)
    InnerBox2 = Box(framelength/2+X, 2*(Y+R2), ESQ_wafer_body,
                    xcent=framelength/2.-(framelength/2.+X)/2.,
                    zcent=bodycenter, voltage=pos_voltage, condid=condidA)
    FrameA = (Frame1-Frame2) + InnerBox1 + InnerBox2

    SOIcenter = zcenter + 0.5*ESQ_wafer_length - ESQ_wafer_si/2  # assume body of SOI is on the left
    Frame1 = Box(framelength, framelength, ESQ_wafer_si,
                 zcent=SOIcenter, voltage=neg_voltage, condid=condidB)
    Frame2 = Box(framelength-2*framewidth, framelength-2*framewidth, 2*ESQ_wafer_si,
                 zcent=SOIcenter, voltage=neg_voltage, condid=condidB)
    InnerBox1 = Box(2*(Y+R2), framelength/2.+X, ESQ_wafer_si,
                    ycent=-framelength/2.+(framelength/2.+X)/2.,
                    zcent=SOIcenter, voltage=neg_voltage, condid=condidB)
    InnerBox2 = Box(2*(Y+R2), framelength/2.+X, ESQ_wafer_si,
                    ycent=framelength/2.-(framelength/2.+X)/2.,
                    zcent=SOIcenter, voltage=neg_voltage, condid=condidB)
    FrameB = (Frame1-Frame2) + InnerBox1 + InnerBox2

    electrodeA = element(voltage=pos_voltage, condid=condidA, rotation=0)
    electrodeB = element(voltage=neg_voltage, condid=condidB, rotation=90)
    electrodeC = element(voltage=pos_voltage, condid=condidA, rotation=180)
    electrodeD = element(voltage=neg_voltage, condid=condidB, rotation=270)

    pos += ESQ_wafer_length
    print("--- ESQ ends at: ", pos)

    return electrodeA + electrodeB + electrodeC + electrodeD + FrameA + FrameB

"""
def RF_stack(voltage, condid, rfgap=200*um):
    #two wafers with grounded planes on the outside and an RF-gap in the middle
    global pos
    print("--- RF starts at: ", pos)

    condidA, condidB, condidC, condidD = condid

    r_beam = 90*um

    SOIcenter = pos + 0.5*ESQ_wafer_si
    Frame = Box(framelength, framelength, 20*um,
                zcent=SOIcenter, voltage=-voltage, condid=condidA)
    Beam = ZCylinder(r_beam, 25*um, zcent=SOIcenter, voltage=voltage, condid=condidA)
    SOI1 = Frame-Beam
    pos += ESQ_wafer_si + ESQ_wafer_box

    bodycenter = pos + 0.5*ESQ_wafer_body
    Frame = Box(framelength, framelength, 500*um,
                zcent=bodycenter, voltage=0, condid=condidB)
    Beam = ZCylinder(r_beam, 510*um, zcent=bodycenter, voltage=0, condid=condidB)
    body1 = Frame-Beam
    pos += ESQ_wafer_body

    print("--- RF gap starts at: ", pos)
    print("--- RF gap length: ", rfgap)
    print("--- RF voltage: ", voltage)
    pos += rfgap
    print("--- RF gap ends at: ", pos)

    bodycenter = pos + 0.5*ESQ_wafer_body
    Frame = Box(framelength, framelength, 500*um,
                zcent=bodycenter, voltage=0, condid=condidC)
    Beam = ZCylinder(r_beam, 510*um, zcent=bodycenter, voltage=0, condid=condidC)
    body2 = Frame-Beam
    pos += ESQ_wafer_body

    pos += ESQ_wafer_box

    SOIcenter = pos + 0.5*ESQ_wafer_si
    Frame = Box(framelength, framelength, 20*um,
                zcent=SOIcenter, voltage=voltage, condid=condidD)
    Beam = ZCylinder(r_beam, 25*um, zcent=SOIcenter, voltage=voltage, condid=condidD)
    SOI2 = Frame-Beam
    pos += ESQ_wafer_si
    print("--- RF ends at: ", pos)

    return SOI1 + body1 + body2 + SOI2
"""

def RF_stack2(condid, rfgap1, rfgap2, rfgap3, voltage=0):
    """two wafers with a gap of rfgap between them. Both wafers are
    insulating with conducting layers on both sides"""

    global pos
    print("--- RF1 starts at: ", pos)

    condidA, condidB, condidC, condidD = condid

    #r_beam = 90*um
    r_beam = 1*mm
    thickness = 2*um

    SOIcenter = pos + 0.5*rfgap1
    Frame = Box(framelength, framelength, rfgap1, voltage=0,
                zcent=SOIcenter, condid=condidA)
    Beam = ZCylinder(r_beam, rfgap1+5*um, zcent=SOIcenter, voltage=0, condid=condidA)
    SOI1 = Frame-Beam
    print("--- RF1 ends at: ", pos)
    #pos += rfgap1 + 500*um
    pos += rfgap1 + 1.5*mm

    print("--- RF2 starts at: ", pos)
    bodycenter = pos + 0.5*rfgap2
    Frame = Box(framelength, framelength, rfgap2, voltage=voltage,
                zcent=bodycenter, condid=condidB)
    Beam = ZCylinder(r_beam, rfgap2+5*um, zcent=bodycenter, voltage=voltage, condid=condidB)
    body1 = Frame-Beam
    print("--- RF2 ends at: ", pos)
    pos += rfgap2 + 500*um

    print("--- RF3 starts at: ", pos)
    SOIcenter = pos + 0.5*rfgap3
    Frame = Box(framelength, framelength, rfgap3, voltage=0,
                zcent=SOIcenter, condid=condidD)
    Beam = ZCylinder(r_beam, rfgap3+5*um, zcent=SOIcenter, voltage=0, condid=condidD)
    SOI2 = Frame-Beam
    pos += rfgap3
    print("--- RF3 ends at: ", pos)

    return SOI1 + body1 + SOI2


def RF_stack3(condid, betalambda_half=200*um, gap=RF_gap, voltage=0):
    """4 wafers rf stack with 2 acceleration stages

     The first two form an RF acceleration cell, followed by a drift
     of length rfgap, followed by a second RF accererlation cell.

    """

    global pos
    condidA, condidB, condidC = condid

    #r_beam = 90*um
    r_beam = 2*mm

    wafercenter = pos + 0.5*RF_thickness
    Frame = Box(framelength, framelength, RF_thickness, voltage=0,
                zcent=wafercenter, condid=condidA)
    Beam = ZCylinder(r_beam, RF_thickness + 50*um, zcent=wafercenter, voltage=0, condid=condidA)
    Ground1 = Frame-Beam
    pos += RF_thickness + gap

    print("middle of first gap " +str(pos-.5*gap))

    length = betalambda_half-gap
    assert length > 0
    bodycenter = pos + 0.5*length
    Frame = Box(framelength, framelength, length, voltage=voltage,
                zcent=bodycenter, condid=condidB)
    Beam = ZCylinder(r_beam, length + 500*um, zcent=bodycenter, voltage=voltage, condid=condidB)
    body1 = Frame-Beam
    pos += length + gap

    print("middle of second gap " +str(pos-.5*gap))

    wafercenter = pos + 0.5*RF_thickness
    Frame = Box(framelength, framelength, RF_thickness, voltage=0,
                zcent=wafercenter, condid=condidC)
    Beam = ZCylinder(r_beam, RF_thickness + 50*um, zcent=wafercenter, voltage=0, condid=condidC)
    Ground2 = Frame-Beam
    pos += RF_thickness

    return Ground1 + body1 + Ground2


def Aperture(voltage, condid, width=10*um):
    """A simple thin wafer with a whole for the beam"""

    global pos
    print("--- Aperture starts at: ", pos)
    print("--- Aperture voltage: ", voltage)
    print("--- Aperture width: ", width)

    r_beam = 90*um
    bodycenter = pos + 0.5*width

    Frame = Box(framelength, framelength, width,
                zcent=bodycenter, voltage=voltage, condid=condid)
    Beam = ZCylinder(r_beam, 2*width, zcent=bodycenter, voltage=voltage, condid=condid)
    body = Frame-Beam

    pos += width
    print("--- Aperture ends at: ", pos)

    return body
