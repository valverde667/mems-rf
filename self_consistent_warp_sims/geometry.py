import warp as wp

# universial dimensions
gapGNDRF = 2 * wp.mm  # those are currently needed in sss_f_t
wafer_thickness = 625 * wp.um
copper_thickness = 35 * wp.um


####
def ESQ(position, invertPolarity, voltage):
    """
    One ESQ wafer
    :param position: of the center of a single ESQ wafer
    :param invertPolarity: inverts polarity,
                            must be 1 or -1
    :return: a single ESQ wafer
    """

    # Dimensions:
    d_beamhole = 1 * wp.mm
    wafer_thickness = 625 * wp.um
    copper_thickness = 35 * wp.um
    d_out_copper = 2 * wp.mm
    quater_gap = 0.25 * wp.mm  # EXPLAIN

    # A this is an Annulus with the
    # dimensions of the PCB for subtraction later.
    pcb = wp.ZAnnulus(
        rmin=d_beamhole / 2 + copper_thickness,
        rmax=10 * wp.mm,
        zcent=position,
        length=wafer_thickness,
    )

    def quarter(
        orientation,
        qvolt=voltage,
        rmin=d_beamhole,
        rmax=d_out_copper,
        zLength=wafer_thickness + 2 * copper_thickness,
        gap=250 * wp.um,
    ):
        """
        orientation:
           3
           y
        0  o x 2
           1
        """
        zsigns = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        rot = 2 * wp.pi / 8
        annulus = wp.ZAnnulus(
            rmin=d_beamhole / 2,
            rmax=d_out_copper / 2,
            length=wafer_thickness + 2 * copper_thickness,
            voltage=qvolt,
            zcent=position,
        )
        planeA = wp.Plane(
            z0=gap,
            zsign=zsigns[orientation][0],
            theta=2 * wp.pi / 4,
            # z-x this stays fixed
            phi=(3 + 2 * orientation) * rot,  # z-y
            voltage=qvolt,
            zcent=0,
        )
        planeB = wp.Plane(
            z0=gap,
            zsign=zsigns[orientation][1],
            theta=2 * wp.pi / 4,
            # z-x this stays fixed
            phi=(5 + 2 * orientation) * rot,  # z-y
            voltage=qvolt,
            zcent=0,
        )
        return annulus - planeA - planeB

    signedV = invertPolarity * voltage
    return (
        quarter(0, signedV)
        + quarter(1, -signedV)
        + quarter(2, signedV)
        + quarter(3, -signedV)
        - pcb
    )


def doublESQ(centerpositions, voltages):
    esqs = None
    for i in range(len(centerpositions)):
        esqoffcenter = (wafer_thickness + copper_thickness) / 2
        print(
            f"ESQ POS at: \n {centerpositions[i] - esqoffcenter}"
            f"\n {centerpositions[i] + esqoffcenter}"
        )
        esqs += ESQ(centerpositions[i] - esqoffcenter, -1, voltages[i]) + ESQ(
            centerpositions[i] + esqoffcenter, -1, voltages[i]
        )
    return esqs


def ESQ_doublet(centerpositions, voltages, volt_ratio, d_wafers=2.695 * wp.mm):
    """
    ESQ doublet, new implementation, April 2020
    timobauer@lbl.com
    :param position: position of the center of the 2 wafers
    :param voltage: applied against ground, wafers are
                    on +/- voltage
    :param d_wafers: gap between the wafers (width of washers)
    :return: an ESQ double
    """

    esqs = None
    for i in range(len(centerpositions)):
        esqoffcenter = d_wafers / 2
        print(
            f"ESQ POS at: \n {centerpositions[i] - esqoffcenter}"
            f"\n {centerpositions[i] + esqoffcenter}"
        )
        esqs += ESQ(centerpositions[i] - esqoffcenter, -1, voltages[i]) + ESQ(
            centerpositions[i] + esqoffcenter, +1, voltages[i] * volt_ratio[i]
        )
    return esqs


def electrical_connections(centerpos, voltage, orientation):
    """
    returns the electrical connections, ether in x or y
    param centerpos: centerposition of the connectiors; NOT the center of the wafer!
    """
    assert orientation in ["x", "y", "xy"]
    # Dimensions:
    d_beamhole = 1 * wp.mm
    copper_thickness = 35 * wp.um
    connector_width = 500 * wp.um
    lattice_constant = 3000 * wp.um

    lattice = wp.Box(
        zcent=centerpos,
        voltage=voltage,
        xsize=lattice_constant + connector_width / 2,
        ysize=lattice_constant + connector_width / 2,
        zsize=copper_thickness,
    ) - wp.Box(
        zcent=centerpos,
        voltage=voltage,
        xsize=lattice_constant - (connector_width / 2),
        ysize=lattice_constant - (connector_width / 2),
        zsize=copper_thickness,
    )

    if orientation == "x":
        xs = lattice_constant
        ys = connector_width
    elif orientation == "y":
        ys = lattice_constant
        xs = connector_width
    elif orientation == "xy":
        return electrical_connections(centerpos, voltage, "y") + electrical_connections(
            centerpos, voltage, "x"
        )
    connectors = wp.Box(
        zcent=centerpos, xsize=xs, ysize=ys, zsize=copper_thickness, voltage=voltage
    ) - wp.ZCylinder(
        zcent=centerpos, radius=d_beamhole / 2, length=copper_thickness, voltage=voltage
    )
    return lattice + connectors


def RF_stack(stackPositions, voltages):
    """This is a rewritten code to make it easier to adapt
    it to the actual teststand"""
    # Defining dimensions:
    d_beamhole = 1 * wp.mm
    wafer_thickness = 625 * wp.um
    copper_thickness = 35 * wp.um
    d_out_copper = 2 * wp.mm
    d_out_copper = 1.5 * wp.mm

    def wafer(centerposition, v):
        """
        A this is a metal hollow zylinder with the
        dimensions of the PCB for subtraction later.
        :param material: should be the same as what it
                        subtracted from
        :return: PCB/silicone part of the wafers
        """

        # The non coducting part of the PCB (the actual board)
        pcb = wp.ZAnnulus(
            rmin=d_beamhole / 2 + copper_thickness,
            rmax=1,  # Basically 'infinite'
            length=wafer_thickness,
            zcent=centerposition,
            voltage=0,
        )
        # Copper part, consisting of a ring in the center and the wires connecting to it
        ring = (
            wp.ZAnnulus(
                rmax=d_out_copper / 2,
                rmin=d_beamhole / 2,
                zcent=centerposition,
                length=wafer_thickness + 2 * copper_thickness,
                voltage=v,
            )
            - pcb
        )
        connectors = electrical_connections(
            centerpos=centerposition + (wafer_thickness / 2 + copper_thickness / 2),
            voltage=v,
            orientation="xy",
        ) + electrical_connections(
            centerpos=centerposition - (wafer_thickness / 2 + copper_thickness / 2),
            voltage=v,
            orientation="xy",
        )
        return connectors + ring

    # Manual overwrite of the positions, always as a
    # centerposition:
    # [[GND, RF, RF, GND],[GND, RF, RF, GND], [..],..]
    # stackPositions = [[.002, .004, .008, .010]]
    stacks = []
    for s, v in zip(stackPositions, voltages):
        stack = wafer(s[0], 0) + wafer(s[1], v) + wafer(s[2], v) + wafer(s[3], 0)
        stacks.append(stack)
    return wp.sum(stacks)


# Energy analyzer prototype setup
# ToDo: This code will be re-written, once a new deflector analyzer design is developed
def spectrometer_v1(
    voltage=1000, d_lastRF_capacitor=25 * wp.mm, d_lastRF_Screen=100 * wp.mm
):
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
    capacitor_Zlength = 15 * wp.mm  # 27.4*wp.mm
    print(f"Placing capacitor at {pos}")
    upperCapacitorPlate = wp.Box(
        0.5 * wp.mm,
        framelength,
        capacitor_Zlength,
        voltage / 2,
        capacitor_plateDist / 2 + capacitor_Xshift,
        0,
        pos,
        condid=301,
        material="Al",
    )
    lowerCapacitorPlate = wp.Box(
        0.5 * wp.mm,
        framelength,
        capacitor_Zlength,
        -voltage / 2,
        -capacitor_plateDist / 2 + capacitor_Xshift,
        0,
        pos,
        condid=302,
        material="Al",
    )

    p1 = wp.Box(
        0.5 * wp.mm,
        framelength,
        0.015,
        voltage=-500,
        xcent=0.001,
        ycent=0,
        zcent=0.05,
        material="Al",
    )
    p2 = wp.Box(
        0.5 * wp.mm,
        framelength,
        0.015,
        voltage=500,
        xcent=-0.001,
        ycent=0,
        zcent=0.05,
        material="Al",
    )
    pos = pos + d_lastRF_Screen
    pos_pos.append(pos)
    print(f"Setting up Scintillator at {pos}")
    return p1 + p2
