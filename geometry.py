"""Module to keep track of different geometries used in our project"""

from warp import *

def ESQ(voltage, zcenter, condid):
    """Simple ESQ wafer

    Use 3 cylinders in Z to make up a single electrode.
    Position at zcenter and use two condids [a,b] for the electrods.

    Bias these +-+- with voltage V.

    """
    R1 = 96*um  # center cylinder
    R2 = 75*um  # outdise cylinders

    length = (500+2+20)*um  # length of a SOI wafer

    X = -337*um  # X offset for outer electrodes
    Y = 125*um  # +-Y offset for outer electrodes
    XX = -187*um  # X offset for inner electrode

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
            print "wrong rotation value"
        zcent1 = zcenter

        electrode1 = ZCylinder(radius=R2, length=length, voltage=voltage,
                                xcent=xcent1, ycent=ycent1, zcent=zcenter,
                                condid=condid)
        electrode2 = ZCylinder(radius=R2, length=length, voltage=voltage,
                                xcent=xcent2, ycent=ycent2, zcent=zcenter,
                                condid=condid)
        electrode3 = ZCylinder(radius=R1, length=length, voltage=voltage,
                                xcent=xcent3, ycent=ycent3, zcent=zcenter,
                                condid=condid)
        return  electrode1 + electrode2 + electrode3

    condidA, condidB = condid
    # frame
    framelength = 1500*um
    framewidth = 150*um

    bodycenter = zcenter - 0.5*length + 250*um  # assume body of SOI is on the left
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

    SOIcenter = zcenter + 0.5*length - 10*um  # assume body of SOI is on the left
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

    return electrodeA + electrodeB + electrodeC + electrodeD + FrameA + FrameB

