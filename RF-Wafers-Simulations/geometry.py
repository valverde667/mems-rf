"""Module to keep track of different geometries used in our current project"""

# 2020-03-23 Timo:
# This is the new geometry file for the sss-for-thread file

import warp as wp
# universial dimensions


# unit cell frame
#framelength = 0.15#\\1500*wp.um
#framewidth = 150*wp.um

#end_accel_gaps = []
#start_accel_gaps = []
#start_ESQ_gaps = []
#end_ESQ_gaps = []
#count = 0 # 1st RF count = 0 grounded, second rf count =1 voltage, third RF count = 2 voltage, fourth RF count =3 grounded, fourth RF count =4 voltage (GVVGGVVGGVVGGVVGG GVVG is being repeated throughout which is why we were doing it this way beforeI'll try to make it work the other way then
#pos_pos =[] #to keep track of what value pos has throughout the making of the RF stack

#should be obsolete by now
# def Gap(dist=500*wp.um): #why is this 500 um, just a default value?
#     """Vacuum gap, e.g. between wafers"""
#     global pos
#     print(f"The position in the Gap function currently is {pos}")
#     pos += dist
#     print(f"After adding {dist} to the pos the pos is now {pos}")
#     print("--- Gap ends at: ", pos)


def ESQ_old(voltage, condid):
    """Simple ESQ wafer

    Use 4 cylinders in Z to make up a single electrode.
    Add to current position(pos) and use two condids [a,b] for the electrodes.

    Bias these +-+- with voltage V.

    """
    global pos
    R1 = (4/7)*wp.mm  #radius of electrodes

    X = (15/14)*wp.mm#2*wp.mm  #X offset for electrodes

    print("--- ESQ starts at: ", mid_gap[-1]) #print("--- ESQ starts at: ", pos)
    start_ESQ_gaps.append(mid_gap[-1]) #array of start of acceleration gaps for future use
    print("--- ESQ voltage: ", voltage)
    zcenter = mid_gap[-1] - 0.5*ESQ_wafer_length #center of ESQ? #+
    print(f"the center of the esq is {zcenter}")

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

    pos =+ mid_gap[-1] + ESQ_gap + .5*ESQ_wafer_length #pos += ESQ_wafer_length
    end_ESQ_gaps.append(pos) #array of end of ESQ gaps for future use
    print("--- ESQ ends at: ", pos)

    return electrodeA + electrodeB + electrodeC + electrodeD
####

def ESQ_double(position, voltage, d_wafers=2*wp.mm):
    '''
    ESQ double, new implementation, April 2020
    timobauer@lbl.com
    :param position: position of the center of the 2 wafers
    :param voltage: applied against ground, wafers are
                    on +/- voltage
    :param d_wafers: gap between the wafers (width of washers)
    :return: an ESQ double
    '''
    # Dimensions:
    d_beamhole = 1 * wp.mm
    wafer_thickness = 625 * wp.um
    copper_thickness = 35 * wp.um
    d_out_copper = 2 * wp.mm
    quater_gap = .25 *wp.mm #EXPLAIN
    #
    #
    def pcb(material):
        return \
            wp.ZCylinder(radius=10 * wp.mm,
                         length=wafer_thickness,
                         xcent=0, ycent=0,
                         zcent=position,
                         material=material,
                         voltage=0
                         ) - wp.ZCylinder(
                radius=d_beamhole / 2 + copper_thickness,
                length=wafer_thickness,
                xcent=0, ycent=0, zcent=position,
                material=material, voltage=0)
    #


###
def RF_stack(stackPositions, voltage):
    """This is a rewritten code to make it easier to adapt
    it to the actual teststand"""
    # Defining dimensions:
    d_beamhole = 1 * wp.mm
    wafer_thickness = 625 * wp.um
    copper_thickness = 35 * wp.um
    d_out_copper = 2 * wp.mm
    #
    def wafer(centerposition, v):
        """This is a single wafer with at a
        centerposition"""
        #
        def pcb(material):
            return \
            wp.ZCylinder(radius=10 * wp.mm,
                         length=wafer_thickness,
                         xcent=0, ycent=0,
                         zcent=centerposition,
                         material=material,
                         voltage=0
                         ) - wp.ZCylinder(
                radius=d_beamhole/2+copper_thickness,
                length=wafer_thickness,
                xcent=0, ycent=0, zcent=centerposition,
                material=material, voltage=0)
        #
        copper = \
            wp.ZCylinder(radius=d_out_copper/2,
                         length=wafer_thickness + 2 *
                                    copper_thickness,
                         xcent=0, ycent=0,
                         zcent=centerposition,
                         material="Cu",
                         voltage=v)\
            - pcb("Cu") \
            - wp.ZCylinder(
                radius=d_beamhole/2,
                length=wafer_thickness +
                2 * copper_thickness,
                xcent=0, ycent=0, zcent=centerposition,
                material="Cu", voltage=v)
        #
        return copper #+pcb("")
    #
    # Manual overwrite of the positions, always as a
    # centerposition:
    # [[GND, RF, RF, GND],[GND, RF, RF, GND], [..],..]
    # stackPositions = [[.002, .004, .008, .010]]
    stacks = []
    for s in stackPositions:
        stack = wafer(s[0], 0) + \
                wafer(s[1], voltage) + \
                wafer(s[2], voltage) + \
                wafer(s[3], 0)
        stacks.append(stack)
    #
    return wp.sum(stacks)


#conductor to absorb particles at a certain Z
# def target_conductor(condid, zcent):
#     #Frame = wp.Box(framelength, framelength, 625*wp.um, voltage=0,zcent=zcent)
#     Frame = wp.Box(framelength, framelength, 1 * wp.mm, voltage=0, zcent=zcent)
#         #target_conductor = wp.ZCylinder(radius=2*wp.mm, length=625*wp.um,
#         #voltage=0, xcent=0, ycent=0, zcent=zcent)
#     return Frame #+ target_conductor

# Spectrometer test setup TODO Clean that up and adapt it to new way of running warp sims
def spectrometer_v1(voltage=1000, d_lastRF_capacitor=25*wp.mm, d_lastRF_Screen=100*wp.mm):
    """
    :param voltage: sets the potential between the two metal plates,
                    they are both charged
    :param d_lastRF_capacitor: distance between the last RF wafer and
                                the center of the capacitor
    :param d_lastRF_Screen: "" to the screen
    :return: the entire spectrometer setup
    """
    # TODO: get rid of pos etc
    global pos
    pos = pos + d_lastRF_capacitor
    pos_pos.append(pos)
    capacitor_Zlength = 15*wp.mm # 27.4*wp.mm
    print(f"Placing capacitor at {pos}")
    upperCapacitorPlate = wp.Box(.5*wp.mm,framelength, capacitor_Zlength,voltage/2,
                                 capacitor_plateDist/2 + capacitor_Xshift, 0, pos, condid=301, material='Al')
    lowerCapacitorPlate = wp.Box(0.5 * wp.mm, framelength, capacitor_Zlength, -voltage / 2,
                                 -capacitor_plateDist / 2 + capacitor_Xshift, 0, pos, condid= 302, material='Al')

    p1 = wp.Box(.5 * wp.mm, framelength, 0.015, voltage=-500,
                                 xcent= 0.001, ycent=0, zcent=0.05, material='Al')
    p2 = wp.Box(.5 * wp.mm, framelength, 0.015, voltage=500,
                                 xcent=-0.001, ycent=0, zcent=0.05, material='Al')
    pos = pos + d_lastRF_Screen
    pos_pos.append(pos)
    print(f"Setting up Scintillator at {pos}")
    return p1 + p2

# should not be needed
# def spectrometer_standalone(voltage=1000,centerposition =
# 40 *wp.mm, distanceplates = 30 *wp.mm):
#     # Dimensions of metal plates
#     plateX = 10 *wp.mm
#     plateY = 50 *wp.mm
#     plateZ = 25 *wp.mm
#     #ToDo add x-shift
#     plate1 = wp.Box(xsize=plateX,ysize=plateY,
#                     zsize=plateZ, xcent=distanceplates/2
#                     , ycent=0, zcent=centerposition,
#                     voltage=voltage/2)
#     plate2 = wp.Box(xsize=plateX, ysize=plateY,
#                     zsize=plateZ, xcent=-distanceplates / 2
#                     , ycent=0, zcent=centerposition,
#                     voltage=-voltage / 2)
#     return plate1 + plate2