import json, os, time
import numpy as np
import scipy.interpolate
import scipy.optimize

pathtocalibdata = "/home/timo/Documents/LBL/Warp/atap-meqalac-simulations/Energy Analyzer/Step3/Calibrations_N3/"
allfiles = []
for f in [9]:  # [6, 8, 9]:
    allfiles.append(f"{pathtocalibdata}{f}/calibrationdata_e.json")


def interpolateddata():
    d = {}
    energy = list()
    voltage = list()
    xscreen = list()
    for file in allfiles:
        with open(file, "r") as fp:
            d = json.load(fp)
        for key in d.keys():
            arr = d[key]
            for i, v in enumerate(arr[0]):
                energy.append(float(key))
                voltage.append(v)
                xscreen.append(arr[1][i][0])
    return energy, voltage, xscreen


def ixscreen(nrg, volt):
    """interpolates xscreen"""
    e, v, x = interpolateddata()

    data = (e, v)
    val = x
    find = (nrg, volt)
    val = scipy.interpolate.griddata(data, val, find)
    print(val)
    return val


def idefvolt(nrg, xscreen=35.865e-3):
    """interpolates deflector voltage"""
    e, v, x = interpolateddata()
    data = (e, x)
    val = v
    find = (nrg, xscreen)
    val = scipy.interpolate.griddata(data, val, find, fill_value=-1)
    if val == -1:
        print(val)
    return val


def ienergy(volts, xscreen=35.865e-3):
    """interpolates deflector voltage"""
    e, v, x = interpolateddata()
    data = (v, x)
    val = e
    find = (volts, xscreen)
    val = scipy.interpolate.griddata(data, val, find, fill_value=-1, method="cubic")
    if val == -1:
        print(val)
    return float(val)


def simshittingtheslit():
    # changing which file to use
    # allfiles = []
    # for f in [6, 7, 8]:
    #    allfiles.append(
    #        f'{pathtocalibdata}{f}/calibrationdata_e.json')
    #
    slitpos = 35.865e-3
    tolerance = 2e-3
    positions = [
        slitpos - tolerance,
        slitpos - tolerance / 2,
        slitpos,
        slitpos + tolerance / 2,
        slitpos + tolerance,
    ]
    outlist = []
    for vol in range(2000, 12000, 100):
        for p in positions:
            nrg = ienergy(vol, p)
            if nrg == -1:
                print("out of interpolation range")
            else:
                outlist.append([nrg, vol])
    return outlist


# e, v, x = interpolateddata()
