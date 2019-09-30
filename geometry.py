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
copper_thickness = 35*wp.um
# esq
ESQ_gap = .7*wp.mm
# globals
pos = 0
mid_gap = []

end_accel_gaps = []
start_accel_gaps = []
start_ESQ_gaps = []
end_ESQ_gaps = []



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
    R1 = (4/7)*wp.mm  #radius of electrodes

    X = (15/14)*wp.mm#2*wp.mm  #X offset for electrodes

    print("--- ESQ starts at: ", pos)
    start_ESQ_gaps.append(pos) #array of start of acceleration gaps for future use
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
    end_ESQ_gaps.append(pos) #array of end of ESQ gaps for future use
    print("--- ESQ ends at: ", pos)

    return electrodeA + electrodeB + electrodeC + electrodeD

def RF_stack3(condid, betalambda_half=200*wp.um, gap=RF_gap, voltage=0):
    """4 wafers rf stack with 2 acceleration stages

     The first two form an RF acceleration cell, followed by a drift
     of length rfgap, followed by a second RF accererlation cell.

    """

    global pos, mid_gap, end_accel_gaps
    condidA, condidB, condidC, condidD = condid

    r_beam = .5*wp.mm #aperature radius

    wafercenter = pos + 0.5*RF_thickness
    print(f"The first wafer center is = {wafercenter}")
    
    r_copper = 1.5*wp.mm
    
    #First RF wafer grounded ------------------------------------------------------------------------------------

    length = betalambda_half-gap #want to use betalambda half to separate the RF cells from each other
    
    
    center_copper_1A = wafercenter-.5*RF_thickness-.5*copper_thickness
    circular_copper_conductor_1A = wp.ZCylinder(radius=r_copper, length=copper_thickness, voltage=0,
                                                 xcent=0, ycent=0, zcent=center_copper_1A, condid=condidA, material='Cu')
    subtraction_beam_1A = wp.ZCylinder(r_beam, length = copper_thickness+.5*wp.mm, zcent=center_copper_1A, voltage=0, condid=condidA, material='Cu')#to be used to subtract from copper conductors
    c_conductor_1A = circular_copper_conductor_1A - subtraction_beam_1A #circular_copper_conductor - subtraction_frame - subtraction_beam
    
    center_copper_1B = wafercenter+RF_thickness+copper_thickness
    circular_copper_conductor_1B = wp.ZCylinder(radius=r_copper, length=copper_thickness, voltage=0,
                                                xcent=0, ycent=0, zcent=center_copper_1B, condid=condidA, material='Cu')
    subtraction_beam_1B = wp.ZCylinder(r_beam, length = copper_thickness+.5*wp.mm, zcent=center_copper_1B, voltage=0, condid=condidA, material='Cu')#to be used to subtract from copper conductors
    c_conductor_1B = circular_copper_conductor_1B - subtraction_beam_1B
    
    #cylinder inbetween two disk conductors, matching Grant's New RF design
    #comment out if not interested in this design
    center_inner_cylinder1 = (center_copper_1A + center_copper_1B )/2
    inner_cylinder_conductor1 = wp.ZCylinder(radius=r_beam + copper_thickness, length=RF_thickness, voltage=0,
                                                xcent=0, ycent=0, zcent=center_inner_cylinder1, condid=condidA, material='Cu')
    subtraction_beam_cylinder1 = wp.ZCylinder(r_beam, length = RF_thickness+.5*wp.mm, zcent=center_inner_cylinder1, voltage=0, condid=condidA, material='Cu')
    inner_cylinder1 = inner_cylinder_conductor1 - subtraction_beam_cylinder1
    
    print(f"the position is currently: {pos}")
    pos += RF_thickness + gap #the position of the end of the acceleration gaps
    print(f"after changing the position it is now: {pos}")
    
    print("middle of first gap " +str(pos-.5*gap))
    mid_gap.append(pos-0.5*gap) #array for middle of acceleration gaps for future use
    end_accel_gaps.append(pos) #array of the end of acceleration gaps for future use
    start_accel_gaps.append(pos-gap) #array of start of acceleration gaps for future use
    
    #Second RF wafer at voltage ---------------------------------------------------------------------------------
    
    print("betalambda_half = {}".format(betalambda_half))
    assert length > 0
    bodycenter = pos + 0.5*length
    print(f"The first body center is = {bodycenter}")
    
    bodycenter_new = wafercenter + gap
    
    center_copper_2A = center_copper_1B +copper_thickness + gap
    circular_copper_conductor_2A = wp.ZCylinder(radius=r_copper, length=copper_thickness, voltage=voltage,
                                                xcent=0, ycent=0, zcent=center_copper_2A, condid=condidB, material='Cu')
    subtraction_beam_2A = wp.ZCylinder(r_beam, length = copper_thickness+.5*wp.mm, zcent=center_copper_2A, voltage=voltage, condid=condidB, material='Cu')#to be used to subtract from copper conductors
    c_conductor_2A = circular_copper_conductor_2A - subtraction_beam_2A #circular_copper_conductor - subtraction_frame - subtraction_beam

    center_copper_2B = center_copper_2A + copper_thickness + RF_thickness
    circular_copper_conductor_2B = wp.ZCylinder(radius=r_copper, length=copper_thickness, voltage=voltage,
                                                xcent=0, ycent=0, zcent=center_copper_2B, condid=condidB, material='Cu')
    subtraction_beam_2B = wp.ZCylinder(r_beam, length = copper_thickness+.5*wp.mm, zcent=center_copper_2B, voltage=voltage, condid=condidB, material='Cu')#to be used to subtract from copper conductors
    c_conductor_2B = circular_copper_conductor_2B - subtraction_beam_2B
    
    #cylinder inbetween two disk conductors, matching Grant's New RF design
    #comment out if not interested in this design
    center_inner_cylinder2 = ( center_copper_2A + center_copper_2B )/2
    inner_cylinder_conductor2 = wp.ZCylinder(radius=r_beam + copper_thickness, length=RF_thickness, voltage=voltage,
                                            xcent=0, ycent=0, zcent=center_inner_cylinder2, condid=condidB, material='Cu')
    subtraction_beam_cylinder2 = wp.ZCylinder(r_beam, length = RF_thickness+.5*wp.mm, zcent=center_inner_cylinder2, voltage=voltage, condid=condidB, material='Cu')
    inner_cylinder2 = inner_cylinder_conductor2 - subtraction_beam_cylinder2
    
    pos += length + gap
    
    print("middle of second gap " + str(pos-.5*gap))
    mid_gap.append(pos-0.5*gap)
    end_accel_gaps.append(pos)
    start_accel_gaps.append(pos-gap) #array of start of acceleration gaps for future use
    
    first_RF_cell = c_conductor_1A + inner_cylinder1 +  c_conductor_1B + c_conductor_2A + inner_cylinder2 + c_conductor_2B
    
    #third RF wafer at ground------------------------------------------------------------------------------------------------

    print(f"The second wafer center is = {wafercenter}")

    center_copper_3A = wafercenter + betalambda_half
    print(f"The third wafer center is = {center_copper_3A}")
    circular_copper_conductor_3A = wp.ZCylinder(radius=r_copper, length=copper_thickness, voltage=voltage,
                                                xcent=0, ycent=0, zcent=center_copper_3A, condid=condidC)
    subtraction_beam_3A = wp.ZCylinder(r_beam, length = copper_thickness+.5*wp.mm, zcent=center_copper_3A, voltage=voltage, condid=condidC)#to be used to subtract from copper conductors
    c_conductor_3A = circular_copper_conductor_3A - subtraction_beam_3A #circular_copper_conductor - subtraction_frame - subtraction_beam
                                                
    center_copper_3B = center_copper_3A+RF_thickness+copper_thickness
    circular_copper_conductor_3B = wp.ZCylinder(radius=r_copper, length=copper_thickness, voltage=voltage,
        xcent=0, ycent=0, zcent=center_copper_3B, condid=condidC)
    subtraction_beam_3B = wp.ZCylinder(r_beam, length = copper_thickness+.5*wp.mm, zcent=center_copper_3B, voltage=voltage, condid=condidC)#to be used to subtract from copper conductors
    c_conductor_3B = circular_copper_conductor_3B - subtraction_beam_3B
    
    #cylinder inbetween two disk conductors, matching Grant's New RF design
    #comment out if not interested in this design
    center_inner_cylinder3 = ( center_copper_3A + center_copper_3B )/2
    inner_cylinder_conductor3 = wp.ZCylinder(radius=r_beam + copper_thickness, length=RF_thickness, voltage=voltage,
                                             xcent=0, ycent=0, zcent=center_inner_cylinder3, condid=condidC, material='Cu')
    subtraction_beam_cylinder3 = wp.ZCylinder(r_beam, length = RF_thickness+.5*wp.mm, zcent=center_inner_cylinder3, voltage=voltage, condid=condidC, material='Cu')
    inner_cylinder3 = inner_cylinder_conductor3 - subtraction_beam_cylinder3
                                                
    print(f"the position is currently: {pos}")
    pos += RF_thickness + gap #the position of the end of the acceleration gaps
    print(f"after changing the position it is now: {pos}")
                                                
    print("middle of first gap " +str(pos-.5*gap))
    mid_gap.append(pos-0.5*gap) #array for middle of acceleration gaps for future use
    end_accel_gaps.append(pos) #array of the end of acceleration gaps for future use
    start_accel_gaps.append(pos-gap) #array of start of acceleration gaps for future use

    #fourth wafer at voltage-----------------------------------------------------------------------------------------
    
    print("betalambda_half = {}".format(betalambda_half))
    assert length > 0
    bodycenter = pos + 0.5*length
    print(f"The first body center is = {bodycenter}")
    
    center_copper_4A = center_copper_3B + gap
    circular_copper_conductor_4A = wp.ZCylinder(radius=r_copper, length=copper_thickness, voltage=0,
                                                xcent=0, ycent=0, zcent=center_copper_4A, condid=condidD)
    subtraction_beam_4A = wp.ZCylinder(r_beam, length = copper_thickness+.5*wp.mm, zcent=center_copper_4A, voltage=0, condid=condidD)#to be used to subtract from copper conductors
    c_conductor_4A = circular_copper_conductor_4A - subtraction_beam_4A #circular_copper_conductor - subtraction_frame - subtraction_beam
                                                
    center_copper_4B = center_copper_4A + copper_thickness + RF_thickness
    circular_copper_conductor_4B = wp.ZCylinder(radius=r_copper, length=copper_thickness, voltage=0,
            xcent=0, ycent=0, zcent=center_copper_4B, condid=condidD)
    subtraction_beam_4B = wp.ZCylinder(r_beam, length = copper_thickness+.5*wp.mm, zcent=center_copper_4B, voltage=0, condid=condidD)#to be used to subtract from copper conductors
    c_conductor_4B = circular_copper_conductor_4B - subtraction_beam_4B
    
    #cylinder inbetween two disk conductors, matching Grant's New RF design
    #comment out if not interested in this design
    center_inner_cylinder4 = ( center_copper_4A + center_copper_4B )/2
    inner_cylinder_conductor4 = wp.ZCylinder(radius=r_beam + copper_thickness, length=RF_thickness, voltage=0,
                                            xcent=0, ycent=0, zcent=center_inner_cylinder4, condid=condidD, material='Cu')
    subtraction_beam_cylinder4 = wp.ZCylinder(r_beam, length = RF_thickness+.5*wp.mm, zcent=center_inner_cylinder4, voltage=0, condid=condidD, material='Cu')
    inner_cylinder4 = inner_cylinder_conductor4 - subtraction_beam_cylinder4
                                                
    pos += length + gap
                                                
    print("middle of second gap " + str(pos-.5*gap))
    mid_gap.append(pos-0.5*gap)
    end_accel_gaps.append(pos)
    start_accel_gaps.append(pos-gap) #array of start of acceleration gaps for future use
                                                
    second_RF_cell = c_conductor_3A + inner_cylinder3 +  c_conductor_3B + c_conductor_4A + inner_cylinder4 + c_conductor_4B
    #--------------------------------------------------------------------------------------------
    
    return first_RF_cell + second_RF_cell


#conductor to absorb particles at a certian Z
def target_conductor(condid, zcent):
    Frame = wp.Box(framelength, framelength, 625*wp.um, voltage=0,zcent=zcent)
        #target_conductor = wp.ZCylinder(radius=2*wp.mm, length=625*wp.um,
        #voltage=0, xcent=0, ycent=0, zcent=zcent)
    return Frame #+ target_conductor
