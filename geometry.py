"""Module to keep track of different geometries used in our project"""

from warp import *

def ESQ(voltage, zcenter, condid):
    """Simple ESQ wafer

    Use 3 cylinders in Z to make up a single electrode.
    Position at zcenter and use two condids [a,b] for the electrods.

    Bias these +-+- with voltage V.
    """
    condidA, condidB = condid

    def element(voltage, rotation):
        """create a single element, rotated X degrees"""

        R1 = 96*um  # center cylinder
        R2 = 75*um  # outdise cylinders

        length = (500+2+20)*um  # length of a SOI wafer

        X = -337*um  # X offset for outer electrodes
        Y = 125*um  # +-Y offset for outer electrodes
        XX = -187*um  # X offset for inner electrode

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
                                condid=condidA)
        electrode2 = ZCylinder(radius=R2, length=length, voltage=voltage,
                                xcent=xcent2, ycent=ycent2, zcent=zcenter,
                                condid=condidA)
        electrode3 = ZCylinder(radius=R1, length=length, voltage=voltage,
                                xcent=xcent3, ycent=ycent3, zcent=zcenter,
                                condid=condidA)
        return  electrode1 + electrode2 + electrode3

    electrodeA = element(voltage=+voltage, rotation=0)
    electrodeB = element(voltage=-voltage, rotation=90)
    electrodeC = element(voltage=+voltage, rotation=180)
    electrodeD = element(voltage=-voltage, rotation=270)

    return electrodeA + electrodeB + electrodeC + electrodeD

