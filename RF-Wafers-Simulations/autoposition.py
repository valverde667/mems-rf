import matplotlib

matplotlib.use("Agg")
from multiprocessing import Pool
import numpy as np
import json, os, threading, time
from scipy.constants import *
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import lmfit

# basepath = "/media/timo/simstore/r7-highrestest/"
basepath = "/home/timo/autoposition/1/"
# simpath = "/home/timo/Documents/LBL/Warp/atap-meqalac-simulations/RF-Wafers-Simulations/single_species_simulation_for_thread.py"
simpath = "/home/timo/autoposition/single_species_simulation_for_thread.py"
# setting up
stepsize = 0.1e-3
steps = 20
dist2 = (2000 + 625 + 35 + 35) * 1e-6 / 2
# setting up run
rf_voltage = 7e3
bunch_length = 1e-9
ekininit = 10e3
freq = 14.8e6  # currently overwritten in def betalambda
tstep = 1e-11
plotsteps = 20
basecommand = (
    f"python3 {simpath}"
    f" --rf_voltage {rf_voltage}"
    f" --bunch_length {bunch_length}"
    f" --ekininit  {ekininit}"
    f" --freq {freq}"
    f" --tstep {tstep}"
    f" --autorun True"
    f" --path {basepath}"
    f" --plotsteps {plotsteps}"
)
#
savegapsprior = 2  # how many gaps to go back for resstoring >=1
savedistancetorfunits = 4e-3  # distance between savespot to rf-gap-center
stopifincluded = True  # Simulation stops if maximum lies within simulated spectrum
#
"""
format of json:
{
    runnumber #### : {
        'rf_voltage' : 7e3, 'bunch_length' : ... , 'ekinmax' : xx, ekinmax10: xx, ekinav: xx,
        'rfgaps' = [[1,2],[5,6],...], 'rfgaps_ideal' : [[],[],..]
    }
}

ID = # RF-UNIT ### runnumber -> 5356 => 5RF Units, run 356
"""


def readjson(ID):
    if type(ID) == int:
        ID = f"{ID:04d}"
    if os.path.isfile(f"{basepath}{ID}.json"):
        with open(f"{basepath}{ID}.json", "r") as readfile:
            data = json.load(readfile)
    else:
        data = {}
    return data


def readalljson(IDa="1000", IDb="9999"):
    """
    Retruns all runs for a given gap (all contiunous sets - same thousands)
    Its not a bug, its a feature
    """
    outdict = {}
    n = int(IDa)
    end = int(IDb)
    print(f"{basepath}{n:04d}.json")
    while os.path.isfile(f"{basepath}{n:04d}.json") and n <= end:
        outdict[f"{n:04d}"] = readjson(n)
        n += 1
    return outdict


def writejson(ID, key, value):
    writedata = readjson(ID)
    writedata[key] = value
    with open(f"{basepath}{ID}.json", "w") as writefile:
        json.dump(writedata, writefile, sort_keys=True, indent=1)


# def resetjson():
#     writedata = {}
#     with open(f'{basepath}data.json', 'w') as writefile:
#         json.dump(writedata, writefile, sort_keys=True, indent=1)


def betalambda(energy, activegap=1):
    rfunit = int(activegap / 2)
    if rfunit < 4:  # adaptation to current design
        f = 14.8e6
    else:
        f = 27e6
    # bl = np.sqrt((ekin + rf_voltage) * elementary_charge / (40 * atomic_mass)) / 2 / f
    bl = np.sqrt((energy) * elementary_charge / (40 * atomic_mass)) / 2 / f
    return bl


# center Array is a 1D array: [gap1, gap2, gap3 ...
def centertoposition(arr):
    posarr = []
    arr.reverse()
    while len(arr):
        a = arr.pop()
        b = arr.pop()
        posarr.append([a - dist2, a + dist2, b - dist2, b + dist2])
    return posarr


# positionarray is a 2d array: [[wafer1,wafer2,w3,4],[w5,w6 ..],..]
def positiontocenter(arr):
    centerarr = []
    for a in arr:
        centerarr.append((a[1] - a[0]) / 2 + a[0])
        centerarr.append((a[3] - a[2]) / 2 + a[2])
    return centerarr


def runcomm(com, x):
    os.system(com)


def runIDs(IDs, limit=20, morecommands=""):
    print(f"starting to run simulation {IDs}")
    time.sleep(1)
    th = []
    for id in IDs:
        c = f'{basecommand} --name "{id}" ' + morecommands
        commands = readjson("0000")["commands"]
        commands.append(c)
        writejson("0000", "commands", commands)
        th.append(threading.Thread(target=runcomm, args=(c, 1)))
        th[-1].start()
        block(limit)
        # th[-1].join() # blocking
        time.sleep(10)  # this timer is here, so should there be an error,
        # they will be printed one after the other. Impact on simulation time is neglectable.


def order(x, y):
    new_x, new_y = zip(*sorted(zip(x, y)))
    return new_x, new_y


def scan(begin, end, stepwidth, activegap, absolute=False):
    s = []
    newIDs = []
    optimalpositions = readjson("0000")["optimalgaps"][
        : activegap - 1
    ]  # this line and the next are now obselete, change of code needed TODO
    assert (
        len(optimalpositions) + 1 == activegap
    )  # check that the next one appends to the right position
    for n in np.arange(begin, end, stepwidth):
        sn = optimalpositions.copy()
        add = optimalpositions[-1] + betalambda(
            ekininit + rf_voltage * activegap, activegap
        )
        if absolute:  # absolute postion and not relative to btalamda/2
            add = 0
        sn.append(add + n)
        if len(sn) % 2 == 1:
            sn.append(sn[-1] + betalambda(ekininit + rf_voltage * len(sn)))
        s.append(sn.copy())
    print(f"run centerpositions : {s}")
    print(f"Setting up {len(s)} new runs")
    #
    alldata = readalljson(f"{activegap * 1000}")
    if alldata != {}:
        number = int(list(alldata.keys())[-1])
        number += 1
    else:
        number = activegap * 1000
    print(f"continuing at number {number}")
    for i in range(len(s)):
        markedpositions = []
        ### Beamsave always Xmm before the gap
        beamsavepositions = np.array(s[i].copy()) - savedistancetorfunits
        beamsavepositions = beamsavepositions.tolist()
        newIDs.append(str(i + number))
        writejson(newIDs[-1], "rf_gaps", centertoposition(s[i].copy()))
        writejson(newIDs[-1], "centerpositions", s[i].copy())
        writejson(newIDs[-1], "beamsavepositions", beamsavepositions)
        print(f"s[{i}] : {s[i]}")
        for p in centertoposition(s[i]):
            print(f" P : {p}")
            for pi in range(1, len(p), 2):
                markedpositions.append(p[pi] + 2.5e-3)
        writejson(newIDs[-1], "markedpositions", markedpositions)
    return newIDs


def energygain(gap):
    optimalenergies = readjson("0000")["optimalenergies"]
    optimalenergies.insert(0, ekininit)
    assert len(optimalenergies) >= gap
    assert gap >= 1
    return optimalenergies[gap] - optimalenergies[gap - 1]


def infoplots():
    optimalenergies = readjson("0000")["optimalenergies"]
    optimalpositions = readjson("0000")["optimalgaps"]
    #
    fs = 15
    fig, ((axEperGap, axBL), (ax3, axratio), (ax5, ax6)) = plt.subplots(
        3, 2, figsize=(15, 15)
    )
    # Plot energy gain per gap and betalamda and actual distance
    gap = []
    gap2 = []
    egain = []
    bl_perfect = []
    bl_lastenergy = []
    actualgapdistance = []
    ratio_bl = []
    d_bl = []
    testarr = []
    tomm = 1e3
    for i in range(len(optimalenergies)):
        gap.append(i + 1)
        egain.append(energygain(i + 1))
        bl_perfect.append(betalambda(optimalenergies[i] + rf_voltage) * tomm)
        bl_lastenergy.append(
            betalambda(optimalenergies[i] + energygain(i + 1)) * tomm
        )  # energy gain for next gap based on old one
    for i in range(len(optimalenergies) - 1):
        gap2.append(i + 1)
        actualgapdistance.append((optimalpositions[i + 1] - optimalpositions[i]) * tomm)
        ratio_bl.append(actualgapdistance[i] / bl_perfect[i])
        d_bl.append(actualgapdistance[i] - bl_perfect[i])
        testarr.append(d_bl[i] / optimalenergies[i] * 1e3)
    print(d_bl)
    axEperGap.plot(gap, egain, marker="o")
    axEperGap.set(title=f"Energy gain per gap", xlabel="Gap", ylabel="Energy gain [eV]")
    axBL.plot(gap, bl_perfect, label="ideal BL", marker="o")
    axBL.plot(gap, bl_lastenergy, label="BL with last energy", marker="o")
    axBL.plot(gap2, actualgapdistance, label="optimzed distance", marker="o")
    axBL.legend()
    axBL.set(
        title="Beta Lambdas", xlabel="# gap", ylabel="[mm]",
    )
    axratio.plot(gap2, ratio_bl, marker="o")
    axratio.set(
        title="ratio of optimzed distance to ideal BL", xlabel="# gap", ylabel="ratio"
    )
    ax3.plot(gap2, d_bl, marker="o")
    ax3.set(title="difference BL ideal to actual", xlabel="# gap", ylabel="[mm]")
    ax5.plot(gap, optimalenergies, marker="o")
    ax5.set(title="Energy after gap", xlabel="# gap", ylabel="[eV]")
    ax6.plot(gap2, testarr)
    ax6.set(title="difference bl / energy", xlabel="# gap", ylabel="[mm]/keV")
    # plt.title(f'Info')
    plt.tight_layout()
    plt.savefig(f"{basepath}info.png", dpi=400)


def correction(gap):
    return (0.45 * (gap) + 1) / 1e3


def optimizepositions(
    startgap,
    endgap,
    ranges=((-0.75e-3, 1e-3), (-0.2e-3, 0.2e-3)),
    parallelsimulations=8,
    maxcpus=10,
):
    """optimzes gaps, ranges gives the number of iterations and their (symmetrical) width"""
    optimalpositions = readjson("0000")["optimalgaps"]
    optimalenergies = readjson("0000")["optimalenergies"]
    print(f"{optimalenergies} : {optimalpositions}")
    if startgap <= len(optimalpositions):
        print(f"WARNING: SIMULATIONS FOR THIS GAP ALREADY EXISTENT")
        time.sleep(10)
    #
    for gap in range(startgap, endgap + 1):
        # Also, instead of calculation a second run could be based on old estimations ToDo
        maximumpos = (
            optimalpositions[gap - 1 - 1]
            + betalambda(optimalenergies[gap - 1 - 1] + rf_voltage, activegap=gap)
            + correction(gap)
        )
        print(f"Expect next maximum at {maximumpos}")
        for r in ranges:
            ids = scan(
                maximumpos + r[0],
                maximumpos + r[1],
                stepwidth=(r[1] - r[0]) / (parallelsimulations),
                activegap=gap,
                absolute=True,
            )
            # beam save / load
            loadfile = ""
            if gap > savegapsprior + 1:
                loadfile = f'--loadbeam "{basepath}{gap - 1}006.json" --beamnumber {gap - 1 - savegapsprior}'

            runIDs(ids, maxcpus, morecommands=loadfile)
            block()
            maximumpos, maximumenergy, maxincluded = findmaximum(gap)
            print(f"FOUND THE FOLLOWING, maximum is included: {maxincluded}.")
            print(optimalpositions)
            print(maximumpos, maximumenergy, maxincluded)
            if len(optimalpositions) == gap:
                optimalpositions[gap - 1] = maximumpos
                optimalenergies[gap - 1] = maximumenergy
            if len(optimalpositions) == gap - 1:
                optimalpositions.append(maximumpos)
                optimalenergies.append(maximumenergy)
            writejson("0000", "optimalgaps", optimalpositions)
            writejson("0000", "optimalenergies", optimalenergies)
            print(optimalpositions)
            # check if maximum is in the checked interval
            if maxincluded and stopifincluded:
                print(f"Maximum is found at {maximumpos}")
                time.sleep(5)
                break
            else:
                print(f"Looking for maximum at {maximumpos}")
                time.sleep(5)
        if gap > 2:
            try:
                infoplots()
                findmaximumplots(gap)
            except:
                pass


def positionenergies(gap):
    """ returns the position, Eaver and Emax lists ordered by position for a gap
        if a pyplot axis is given, it plots the energies """
    rj = readalljson(f"{gap * 1000}")
    ids = rj.keys()
    print(f"Found IDs {ids} for gap {gap}")
    centerpositions = []
    markedenergies = []
    for id in ids:
        centerpositions.append(rj[id]["centerpositions"])
        markedenergies.append(rj[id]["markedpositionsenergies"])
    # print(f'markedenergies : {markedenergies}')
    # print(f'centerpositions : {centerpositions}')
    pos = []
    eav = []
    emax = []
    #
    # print(f'mes: {mes}')
    for j, me in enumerate(markedenergies):
        me = me[gap - 1]
        # print(f'me: {me}')
        pos.append(centerpositions[j][gap - 1])
        eav.append(me["ekinav"])
        emax.append(me["ekinmax"])
    opos, oeav = order(pos.copy(), eav.copy())
    opos, oemax = order(pos.copy(), emax.copy())
    return opos, oeav, oemax


def singleRFplot(gap):
    pos, eav, emax = positionenergies(gap)
    fig, ax = plt.subplots()

    ax.plot(np.array(pos) * 1e3, np.array(eav) * 1e-3, marker="o")
    plt.title(f"Scan of gap {gap}.")
    plt.xlabel("[mm]")
    plt.ylabel("Energy [keV]")
    #
    plt.show()
    plt.close()


def findmaximum(gap, axis=False):
    """ checks if a local maximum is found. If so, returns it"""
    pos, eav, emax = positionenergies(gap)
    # first check if we already found a local maximum
    energy = eav  # or emax
    maximumE = max(energy)
    indexmaximumE = energy.index(maximumE)
    if indexmaximumE == 0:
        maximumincluded = False  # the search needs to continue to the left
    if indexmaximumE == len(energy) - 1:
        maximumincluded = False  # the search needs to continue to the right
    #
    print(f" Maximal energy from list : {maximumE}")
    # ok, we have a local maximum
    maximumincluded = True

    def quadmodel(x, x0, a, max):
        return -a * ((x - x0) ** 2) + max

    model = lmfit.Model(quadmodel)
    result = model.fit(energy, x=pos, x0=pos[indexmaximumE], a=5e8, max=maximumE)
    print(result.fit_report())
    print(result.best_values)
    maximumEposition = result.best_values["x0"]
    print(f"new optimal position: {maximumEposition}")
    #
    if axis:
        axis.plot(np.array(pos), np.array(eav), marker="o", linestyle="None")
        axis.plot(
            [maximumEposition, maximumEposition],
            [maximumE + 10, maximumE - 100],
            color="black",
            alpha=0.5,
            label=f"{maximumEposition * 1e3:0.2f}mm",
        )
        axis.plot(np.array(pos), result.best_fit, label="Best fit")
        axis.plot(np.array(pos), result.init_fit, "k--", label="initial fit")
        axis.legend()
        axis.set(
            title=f"Scan of gap {gap}.\n{result.best_values['max']:0.2f}eV at {result.best_values['x0'] * 1e3:0.2f}mm",
            xlabel="[m]",
            ylabel="Energy [eV]",
        )
    # return position of maximum and its value
    return maximumEposition, result.best_values["max"], maximumincluded


def findmaximumplots(highestgap):
    fig, axs = plt.subplots(highestgap - 1, figsize=(12, 12 * (highestgap - 1)))
    for gap in range(2, highestgap + 1):
        maximumpos, maximumenergy, maxincluded = findmaximum(gap, axs[gap - 2])
    plt.tight_layout()
    plt.savefig(f"{basepath}findingmaximum.png", dpi=300)


def block(maxthreads=1):
    while threading.active_count() > maxthreads:
        print(f"Active Simulations :" f" {threading.active_count() - 1}")
        time.sleep(1)


def rbp(run, plotnr):
    runIDs(run)
    block()
    singleRFplot(plotnr)


#
# pos, eav, emax = singleRFplot(3)
# findcenter(pos, eav)
