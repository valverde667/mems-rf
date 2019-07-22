"""Module to keep track of different geometries used in our current project"""
import warp as wp
# universial dimensions

# unit cell frame
framelength = 0.15#\\1500*wp.um
framewidth = 150*wp.um

# wafer dimenions
# SOI
ESQ_wafer_length = 625*wp.um #
# RF
RF_thickness = 625*wp.um
RF_gap = 2000*wp.um #acceleration gap
# esq
ESQ_gap = .7*wp.mm #how is this calculated?
# globals
pos = 0
mid_gap = []

def Gap(dist=500*wp.um): #why is this 500 um, just a default value?
    """Vacuum gap, e.g. between wafers"""
    global pos
    pos += dist
    print("--- Gap ends at: ", pos)


def ESQ(voltage, condid):
    """Simple ESQ wafer

    Use 4 cylinders in Z to make up a single electrode.
    Add to current position(pos) and use two condids [a,b] for the electrods.

    Bias these +-+- with voltage V.

    """
    global pos
    R1 = .5*(8/7)*wp.mm  #radius of electrodes

    X = .5*(15/7)*wp.mm#2*wp.mm  #X offset for electrodes

    print("--- ESQ starts at: ", pos)
    print("--- ESQ voltage: ", voltage)
    zcenter = pos + 0.5*ESQ_wafer_length

    def element(voltage, condid, rotation):
        """create a single element, rotated X degrees"""

        if rotation == 0:
            xcent1, ycent1 = X, 0
        elif rotation == 90:
            xcent1, ycent1 = 0, -X
        elif rotation == 180:
            xcent1, ycent1 = -X, 0
        elif rotation == 270:
            xcent1, ycent1 = 0, X
        else:
            print("wrong rotation value")
        
        electrode = wp.ZCylinder(radius=R1, length=ESQ_wafer_length, voltage=voltage,
                               xcent=xcent1, ycent=ycent1, zcent=zcenter,
                               condid=condid)
        return electrode

    condidA, condidB = condid

    # cylinders
    if callable(voltage):
        pos_voltage = voltage

        def neg_voltage(x):
            return -voltage(x)
    else:
        pos_voltage = voltage
        neg_voltage = -voltage

    electrodeA = element(voltage=pos_voltage, condid=condidA, rotation=0)
    electrodeB = element(voltage=neg_voltage, condid=condidB, rotation=90)
    electrodeC = element(voltage=pos_voltage, condid=condidA, rotation=180)
    electrodeD = element(voltage=neg_voltage, condid=condidB, rotation=270)

    pos += ESQ_wafer_length
    print("--- ESQ ends at: ", pos)

    return electrodeA + electrodeB + electrodeC + electrodeD

def RF_stack3(condid, betalambda_half=200*wp.um, gap=RF_gap, voltage=0):
    """4 wafers rf stack with 2 acceleration stages

     The first two form an RF acceleration cell, followed by a drift
     of length rfgap, followed by a second RF accererlation cell.

    """

    global pos, mid_gap
    condidA, condidB, condidC = condid

    r_beam = .5*wp.mm #aperature radius -MWG

    wafercenter = pos + 0.5*RF_thickness
    Frame = wp.Box(framelength, framelength, RF_thickness, voltage=0,
                zcent=wafercenter, condid=condidA)
                #why is there 50um being added to the rf thickness below?
    Beam = wp.ZCylinder(r_beam, RF_thickness + 50*wp.um, zcent=wafercenter, voltage=0, condid=condidA)
    Ground1 = Frame-Beam
    pos += RF_thickness + gap

    print("middle of first gap " +str(pos-.5*gap))
    mid_gap.append(pos-0.5*gap) #-MWG this creates an array that holds all of the middle of the gaps for future use
    
    length = betalambda_half-gap
    print("betalambda_half = {}".format(betalambda_half))
    assert length > 0
    bodycenter = pos + 0.5*length
    Frame = wp.Box(framelength, framelength, length, voltage=voltage,
                zcent=bodycenter, condid=condidB)
    Beam = wp.ZCylinder(r_beam, length + 500*wp.um, zcent=bodycenter, voltage=voltage, condid=condidB)
    body1 = Frame-Beam
    pos += length + gap

    print("middle of second gap " +str(pos-.5*gap))
    mid_gap.append(pos-0.5*gap)
    
    wafercenter = pos + 0.5*RF_thickness
    Frame = wp.Box(framelength, framelength, RF_thickness, voltage=0,
                zcent=wafercenter, condid=condidC)
    Beam = wp.ZCylinder(r_beam, RF_thickness + 50*wp.um, zcent=wafercenter, voltage=0, condid=condidC)
    Ground2 = Frame-Beam
    pos += RF_thickness

    return Ground1 + body1 + Ground2
