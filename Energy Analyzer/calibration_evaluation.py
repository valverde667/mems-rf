import json, os, glob
import numpy as np
import matplotlib.pyplot as plt

pathtosims = "/home/timo/Documents/LBL/Warp/atap-meqalac-simulations/Spectrometer-Sim/step2/Calibrations_N3/"
pathtooutput = '/home/timo/Documents/LBL/Warp/atap-meqalac' \
               '-simulations/Spectrometer-Sim/Step3' \
               '/Calibrations_N3/'


def byEnergy(nrg):
    energies = list()
    voltages = list()
    xScreen = list()
    files = glob.glob(
        os.path.join(pathtosims,
                     f"{nrg:.1f}"
                     f"eV_*V_calibration.json"))
    for f in files:
        with open(f, 'r') as fp:
            arr = json.load(fp)
            if arr['x'] != []:
                energies.append(arr['energy'])
                voltages.append(arr['voltage'])
                xScreen.append(arr['x'])
    return energies, voltages, xScreen


def byVoltage(volt):
    voltages = list()
    xScreen = list()
    energies = list()
    files = glob.glob(
        os.path.join(pathtosims,
                     f"*eV_{volt:.1f}V_calibration.json"))
    print(files)
    for f in files:
        with open(f, 'r') as fp:
            arr = json.load(fp)
            if arr['x']:
                energies.append(arr['energy'])
                voltages.append(arr['voltage'])
                xScreen.append(arr['x'][0])
    return energies, voltages, xScreen

####
#listOfVoltages = range(0, 29000, 1000)
listOfVoltages = range(1000, 12000, 50)
listOfEnergies = range(1000, 24000, 50)
linerange = [2500, 22500]
####
outarr = dict()
plt.figure(figsize=(25,20))
fig, ax = plt.subplots()
# Adding lines
ax.plot(linerange, [35.865,35.865],
        color='black', alpha=0.3)
ax.plot(linerange, [35.865 - .5,35.865 - .5],
        color='black', alpha=0.3)
ax.plot(linerange, [35.865 + .5,35.865 + .5],
        color='black', alpha=0.3)
#
for v in listOfVoltages:
    en, vo, xS = byVoltage(v)
    #
    tosort = list()
    for i, ee in enumerate(en):  # moving them together
        tosort.append([ee, xS[i]])
    tosort = tosort.copy()
    tosort.sort()  # sorting
    en = []
    xS = []
    for i, k in enumerate(tosort):  # and sperating it
        en.append(k[0])
        xS.append(k[1])
    #
    outarr[v] = [en, xS]
    f = f'{pathtooutput}calibrationdata_v.json'
    with open(f, 'w') as fp:
        json.dump(outarr, fp)
    #
    if v == 22000:
        print('xxxxxxxxxxxxxx')
        print(en)
        print('xxxxxxxxx')
        print(xS)
    if en and xS:
        ax.plot(en, np.array(xS) * 1e3, label=f'{v} V')
        ax.text(en[-1], np.array(xS)[-1] * 1e3, f'{v} keV',
                rotation=-35, fontsize=3,
                rotation_mode="anchor")
    # ax.scatter(en, np.array(xS) * 1e3, label=f'{v} V',
    #            marker='x')
    #
# plt.ylim(-.1, 1)
# plt.xlim(0, 100)
# ax.scatter([7000, 7000, 7000], [35.865, 35.865 + .5,
#                                 35.865 - .5], marker="_",
#            color='black')

plt.legend(loc=1, prop={'size': 3})
ax.set(xlabel="Energy (keV)", ylabel='Screen x (mm)',
       title='Calibration of Spectrometer - 1')
plt.grid(b=True, which="major")
# plt.yticks(range(0,95,5))
# plt.yticks(np.arange(25,55,1))
# plt.xticks(range(6000,7600,50), rotation='vertical')
plt.tight_layout()
plt.savefig(f'{pathtooutput}calibration_plot_v',dpi = 500)
#plt.show()
plt.close()

linerange = [3000,12000]
#listOfEnergies = [7000]
outarr = dict()
plt.figure(figsize=(25,20))
fig, ax = plt.subplots()
# Adding lines
ax.plot(linerange, [35.865,35.865],
        color='black', alpha=0.3)
ax.plot(linerange, [35.865 - .5,35.865 - .5],
        color='black', alpha=0.3)
ax.plot(linerange, [35.865 + .5,35.865 + .5],
        color='black', alpha=0.3)
#
for e in listOfEnergies:
    en, vo, xS = byEnergy(e)
    #
    tosort = list()
    for i, vv in enumerate(vo):  # moving them together
        tosort.append([vv, xS[i]])
    tosort = tosort.copy()
    tosort.sort()  # sorting
    vo = []
    xS = []
    for i, k in enumerate(tosort):  # and sperating it
        vo.append(k[0])
        xS.append(k[1])
    #
    outarr[e] = [vo, xS]
    f = f'{pathtooutput}calibrationdata_e.json'
    with open(f, 'w') as fp:
        json.dump(outarr, fp)
    #
    if vo and xS:
        ax.plot(vo, np.array(xS) * 1e3, label=f'{e} keV')
        ax.text(vo[-1], np.array(xS)[-1]*1e3, f'{e} keV',
                rotation=45, fontsize=3,
                rotation_mode="anchor")
#
# plt.ylim(-.1, 1)
# plt.xlim(0, 23000)
# plt.legend(loc=2, prop={'size': 2})
ax.set(xlabel="Cap V", ylabel='Screen x (mm)',
       title='Calibration of Spectrometer')
plt.grid(b=True, which="major")
# plt.yticks(range(25,55,1))
# plt.xticks(range(2900,4600,50), rotation="vertical")
plt.tight_layout()
plt.savefig(f'{pathtooutput}calibration_plot_e',dpi =
1000)
#plt.show()
plt.close()

def getAngle(nrg, volt):
    f = f'{pathtosims}{nrg:.1f}eV_' \
        f'{volt:.1f}V_calibration.json'
    with open(f, 'r') as fp:
        dic = json.load(fp)
        alpha = np.arctan(dic["vx"]/dic['vz']) #in rad
    return alpha/(2*np.pi)  * 360

def simulationshittingtheslit():
    max = 35.86+10
    min = 35.86-10
    acceptedPairs = []
    files = glob.glob(
        os.path.join(pathtosims,
                     f"*_calibration.json"))
    for f in files:
        with open(f, 'r') as fp:
            dic = json.load(fp)
        print(dic['x'][0])
        if min < dic['x'][0] * 1000 < max:
            acceptedPairs.append([dic['energy'],
                                  dic['voltage']])
    print(acceptedPairs)

#simulationshittingtheslit()