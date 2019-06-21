"""
    Geometry to generate the 1/4 wavelength structure
    Created by Grace Woods on 4/1/2018
"""
from warp import *

# unit cell frame
framelength = 5e-3
framewidth = .150e-3

# RF
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


def quart_stack(condid , r_aperture = 1e-3, rfgap=1.5e-3 , drift=0.7e-3 , voltage=0):
    global pos
    condidA, condidB, condidC, condidD = condid

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
