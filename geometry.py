"""Module to keep track of different geometries used in our project"""
import warp as wp
# universial dimensions

# unit cell frame
framelength = 0.15#\\1500*wp.um
framewidth = 150*wp.um

# wafer dimenions
# SOI
ESQ_wafer_body = 678*wp.um #500*wp.um
ESQ_wafer_box = 2*wp.um
ESQ_wafer_si = 20*wp.um
ESQ_wafer_length = ESQ_wafer_body + ESQ_wafer_box + ESQ_wafer_si
# RF
RF_thickness = 625*wp.um
RF_gap = 2000*wp.um  #5000*wp.um #acceleration gap
# esq
ESQ_gap = .7*wp.mm
# globals
pos = 0


def Gap(dist=500*wp.um): #why is this 500 um?
    """Vacuum gap, e.g. between wafers"""
    global pos
    pos += dist
    print("--- Gap ends at: ", pos)


def ESQ(voltage, condid):
    """Simple ESQ wafer

    Use 3 cylinders in Z to make up a single electrode.
    Add to current position(pos) and use two condids [a,b] for the electrods.

    Bias these +-+- with voltage V.

    """
    global pos
    scaling_factor = 10 #10 
    R1 = 1*wp.mm #96*wp.um*scaling_factor  # center cylinder
    R2 = 75*wp.um*scaling_factor  # outside cylinders

    X = -337*wp.um*scaling_factor  # X offset for outer electrodes
    Y = 125*wp.um*scaling_factor  # +-Y offset for outer electrodes
    XX = 2*wp.mm  #-187*wp.um*scaling_factor  # X offset for inner electrode

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

        electrode1 = wp.ZCylinder(radius=R2, length=ESQ_wafer_length, voltage=voltage,
                               xcent=xcent1, ycent=ycent1, zcent=zcenter,
                               condid=condid)
        electrode2 = wp.ZCylinder(radius=R2, length=ESQ_wafer_length, voltage=voltage,
                               xcent=xcent2, ycent=ycent2, zcent=zcenter,
                               condid=condid)
        electrode3 = wp.ZCylinder(radius=R1, length=ESQ_wafer_length, voltage=voltage,
                               xcent=xcent3, ycent=ycent3, zcent=zcenter,
                               condid=condid)
        return electrode3#electrode1 + electrode2 + electrode3

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
    Frame1 = wp.Box(framelength, framelength, ESQ_wafer_body,
                 zcent=bodycenter, voltage=pos_voltage, condid=condidA)
    Frame2 = wp.Box(framelength-2*framewidth, framelength-2*framewidth, ESQ_wafer_body*1.1,
                 zcent=bodycenter, voltage=pos_voltage, condid=condidA)
    InnerBox1 = wp.Box(framelength/2+X, 2*(Y+R2), ESQ_wafer_body,
                    xcent=-framelength/2.+(framelength/2.+X)/2.,
                    zcent=bodycenter, voltage=pos_voltage, condid=condidA)
    InnerBox2 = wp.Box(framelength/2+X, 2*(Y+R2), ESQ_wafer_body,
                    xcent=framelength/2.-(framelength/2.+X)/2.,
                    zcent=bodycenter, voltage=pos_voltage, condid=condidA)
    FrameA = (Frame1-Frame2) + InnerBox1 + InnerBox2

    SOIcenter = zcenter + 0.5*ESQ_wafer_length - ESQ_wafer_si/2  # assume body of SOI is on the left
    Frame1 = wp.Box(framelength, framelength, ESQ_wafer_si,
                 zcent=SOIcenter, voltage=neg_voltage, condid=condidB)
    Frame2 = wp.Box(framelength-2*framewidth, framelength-2*framewidth, 2*ESQ_wafer_si,
                 zcent=SOIcenter, voltage=neg_voltage, condid=condidB)
    InnerBox1 = wp.Box(2*(Y+R2), framelength/2.+X, ESQ_wafer_si,
                    ycent=-framelength/2.+(framelength/2.+X)/2.,
                    zcent=SOIcenter, voltage=neg_voltage, condid=condidB)
    InnerBox2 = wp.Box(2*(Y+R2), framelength/2.+X, ESQ_wafer_si,
                    ycent=framelength/2.-(framelength/2.+X)/2.,
                    zcent=SOIcenter, voltage=neg_voltage, condid=condidB)
    FrameB = (Frame1-Frame2) + InnerBox1 + InnerBox2

    electrodeA = element(voltage=pos_voltage, condid=condidA, rotation=0)
    electrodeB = element(voltage=neg_voltage, condid=condidB, rotation=90)
    electrodeC = element(voltage=pos_voltage, condid=condidA, rotation=180)
    electrodeD = element(voltage=neg_voltage, condid=condidB, rotation=270)

    pos += ESQ_wafer_length
    print("--- ESQ ends at: ", pos)

    return electrodeA + electrodeB + electrodeC + electrodeD# + FrameA + FrameB

def RF_stack3(condid, betalambda_half=200*wp.um, gap=RF_gap, voltage=0):
    """4 wafers rf stack with 2 acceleration stages

     The first two form an RF acceleration cell, followed by a drift
     of length rfgap, followed by a second RF accererlation cell.

    """

    global pos
    condidA, condidB, condidC = condid

    r_beam = .5*wp.mm #90*wp.um*2 #aperature radius -MWG

    wafercenter = pos + 0.5*RF_thickness
    Frame = wp.Box(framelength, framelength, RF_thickness, voltage=0,
                zcent=wafercenter, condid=condidA)
    Beam = wp.ZCylinder(r_beam, RF_thickness + 50*wp.um, zcent=wafercenter, voltage=0, condid=condidA)
    Ground1 = Frame-Beam
    pos += RF_thickness + gap

    print("middle of first gap " +str(pos-.5*gap))

    length = betalambda_half-gap
    assert length > 0
    bodycenter = pos + 0.5*length
    Frame = wp.Box(framelength, framelength, length, voltage=voltage,
                zcent=bodycenter, condid=condidB)
    Beam = wp.ZCylinder(r_beam, length + 500*wp.um, zcent=bodycenter, voltage=voltage, condid=condidB)
    body1 = Frame-Beam
    pos += length + gap

    print("middle of second gap " +str(pos-.5*gap))

    wafercenter = pos + 0.5*RF_thickness
    Frame = wp.Box(framelength, framelength, RF_thickness, voltage=0,
                zcent=wafercenter, condid=condidC)
    Beam = wp.ZCylinder(r_beam, RF_thickness + 50*wp.um, zcent=wafercenter, voltage=0, condid=condidC)
    Ground2 = Frame-Beam
    pos += RF_thickness

    return Ground1 + body1 + Ground2


def Aperture(voltage, condid, width=10*wp.um):
    """A simple thin wafer with a whole for the beam"""

    global pos
    print("--- Aperture starts at: ", pos)
    print("--- Aperture voltage: ", voltage)
    print("--- Aperture width: ", width)

    r_beam = 90*wp.um
    bodycenter = pos + 0.5*width

    Frame = wp.Box(framelength, framelength, width,
                zcent=bodycenter, voltage=voltage, condid=condid)
    Beam = wp.ZCylinder(r_beam, 2*width, zcent=bodycenter, voltage=voltage, condid=condid)
    body = Frame-Beam

    pos += width
    print("--- Aperture ends at: ", pos)

    return body
