from multiprocessing import Pool
import numpy as np
import json, os, threading, time
from scipy.constants import *
from matplotlib import pyplot as plt

basepath = '/media/timo/simstore/r1/'
simpath = '/home/timo/Documents/LBL/Warp/atap-meqalac-simulations/RF-Wafers-Simulations/single_species_simulation_for_thread.py'
# setting up
stepsize = 0.1e-3
steps = 8
dist2 = (4000 + 625 + 35 + 35) * 1e-6 / 2
# setting up run
rf_voltage = 7e3
bunch_length = 1e-9
ekininit = 10e3
freq = 14.8e6
tstep = 7e-11
basecommand = f'python3 {simpath}' \
              f' --rf_voltage {rf_voltage}' \
              f' --bunch_length {bunch_length}' \
              f' --ekininit  {ekininit}' \
              f' --freq {freq}' \
              f' --tstep {tstep}' \
              f' --autorun True' \
              f' --path {basepath}'
#
'''
format of json:
{
    runnumber #### : {
        'rf_voltage' : 7e3, 'bunch_length' : ... , 'ekinmax' : xx, ekinmax10: xx, ekinav: xx,
        'rfgaps' = [[1,2],[5,6],...], 'rfgaps_ideal' : [[],[],..]
    }
}

ID = # RF-UNIT ### runnumber -> 5356 => 5RF Units, run 356
'''


def readjson():
    with open(f'{basepath}data.json', 'r') as readfile:
        data = json.load(readfile)
    return data


def writejson(ID, key, value):
    writedata = readjson()
    if ID not in writedata.keys():
        writedata[ID] = {}
    writedata[ID][key] = value
    with open(f'{basepath}data.json', 'w') as writefile:
        json.dump(writedata, writefile, sort_keys=True, indent=1)


def resetjson():
    writedata = {}
    with open(f'{basepath}data.json', 'w') as writefile:
        json.dump(writedata, writefile, sort_keys=True, indent=1)


def betalambda(ekin):
    bl = np.sqrt((ekin + rf_voltage) * elementary_charge / (40 * atomic_mass)) / 2 / freq
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


def runIDs(IDs):
    print(f'starting to run simulation {IDs}')
    th = []
    for id in IDs:
        c = f'{basecommand} --name "{id}"'
        th.append(threading.Thread(target=runcomm, args=(c, 1)))
        th[-1].start()
        th[-1].join()
    while threading.active_count() > 1:
        print(f'Active Simulations :'
              f' {threading.active_count() - 1}')
        time.sleep(1)


def step1_firstpositions():
    # firstpositions = [[.0036525, .0056525, 0.01323279, 0.01523279]]
    firstcp = [4.7e-3, 4.7e-3 + betalambda(ekininit + rf_voltage)]
    s = []
    for n in np.arange(-5, -5 + steps, 1):
        s.append([firstcp[0] + (n * stepsize), firstcp[1] + (n * stepsize)])
    #
    print(f's : {s}')
    for i in np.arange(0, steps):
        markedpositions = []
        writejson(str(i + 1000), "rf_gaps", centertoposition(s[i].copy()))
        writejson(str(i + 1000), "centerpositions", s[i].copy())
        print(f's[{i}] : {s[i]}')
        for p in centertoposition(s[i]):
            markedpositions.append(p[1] + 1e-3)
        writejson(str(i + 1000), "markedpositions", markedpositions)


def step2_optimizepositions():
    '''Running the sim on last 8 (steps) IDs'''
    data = readjson()
    keys = data.keys()
    runIDs(keys)


idx = ["1000", "1001", "1002", "1003", "1004", "1005", "1006", "1007"]


def singleRFplot():
    ids = idx
    rj = readjson()
    centerpositions = []
    markedenergies = []
    for id in ids:
        centerpositions.append(rj[id]["centerpositions"])
        markedenergies.append(rj[id]["markedpositionsenergies"])
    print(f'markednrg : {markedenergies}')
    pos = []
    eav = []
    emax = []
    for i, mes in enumerate(markedenergies):
        print(f'mes: {mes}')
        for j, me in enumerate(mes):
            print(f'me: {me}')
            pos.append(centerpositions[i][j] * 1e3)
            eav.append(me["ekinav"] / 1e3)
            emax.append(me["ekinmax"] / 1e3)
            #
            print(pos)
            print(eav)
    fig, ax = plt.subplots()
    ax.plot(pos, eav)
    plt.show()


def runall():
    resetjson()
    step1_firstpositions()
    step2_optimizepositions()
