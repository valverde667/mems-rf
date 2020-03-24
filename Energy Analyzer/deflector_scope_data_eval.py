import json, os, csv, time
import matplotlib.pyplot as plt
import numpy as np
import interpolation

deffolderinp = \
    '/home/timo/Documents/LBL/DataEval/2020-03-11/'
deffolderout = '/home/timo/Documents/LBL/DataEval/2020-03-11/'

try:
    os.mkdir(deffolderout)
except:
    pass


##############  THIS IS THE EVALUATION OF 2020-03-11

def loadcsv(filename):
    '''IDK IF THIS WORKS'''
    data = []
    with open(f'{deffolderinp}{filename}', 'r') as fp:
        reader = csv.reader(fp, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                firstrow = row
            if i != 1 and i != 0:
                data.append(row)
    return firstrow, data[::100]


def nIons(timespan, time, volts):
    t = np.array(time) * 1e6
    v = np.array(volts)
    out = 0
    c = 0
    for i, t in enumerate(t):
        if timespan[0] < t and t < timespan[1]:
            c += 1
            out += v[i]
    # out = out / c / (
    # timespan[1] - timespan[0])  # normalizing
    # print(f'nions out : {out}')
    return out


def rowfileplot(chs=[1], title='', ):
    slitpos = 35.865e-3
    widthslit  = .5e-3
    rfvlots = [5, 10, 20, 40, 60]
    defvoltsarr = [
        list(range(1700, 2125, 25)),
        list(range(1700, 2125, 25)),
        list(range(1650, 2200, 25)),
        list(range(1500, 2450, 50)),
        list(range(1400, 2650, 50)),
    ]
    #
    collectedx = []
    collectedy = []
    #

    for n, rfv in enumerate(rfvlots):
        defvolts = defvoltsarr[n]
        energylist = []
        # add error
        for i in defvolts:
            energylist.append(
                interpolation.ienergy(i * 2, slitpos))
            # print(i,energylist[-1])
        if -1 in energylist:
            print("PROBLEM")
        # print(defvolts)
        # print('ENERGYLIST')
        # print(energylist)
        nI = []
        for ch in chs:
            for j, gn in enumerate(
                    range(0, defvolts.__len__())):
                t = []
                ms = []
                results = []
                with open(
                        f"{deffolderinp}{rfv}V/{rfv}V_RF_{gn}.csv") as csvfile:
                    reader = csv.reader(
                        csvfile)
                    for row in reader:  # each row is a list
                        results.append(row)
                    # select specific time
                    for i in results[2:]:
                        t.append(float(i[0]))
                        ms.append(float(i[ch]))
                fig, ax = plt.subplots()
                ax.plot(np.array(t) * 1e6,
                        np.array(ms) * 1e3,
                        linewidth=0.1)
                plt.ylim(-5, 220)
                ax.set(xlabel='time (us)', ylabel=f'mV')
                plt.title(
                    f'deflector Voltage: {defvolts[j]}\n'
                    f'Energy : {energylist[j] / 1e3:02.3f} keV\n'
                    f'RF amp Voltage: {rfv}V'
                )
                # plt.yticks(np.arange(-65, 65, 5))
                plt.grid(b=True)
                # plt.savefig(
                #    f'{deffolderout}{rfv}V/05V_ch{ch}_{defvolts[j]}V_{energylist[j] / 1e3:02.3f}_keV.png',
                #    dpi=300)
                plt.close()
                # for plot of deflection/energy vs npaticles

                nI.append(nIons([175, 1500], t, ms))
                # print(energylist[j],nI[-1])
        ###
        fig, ax = plt.subplots()
        # ax.scatter(energylist, nI, marker='x')
        print(
            f'plotting length {defvolts.__len__()} :: {nI.__len__()}')
        # ax.plot(defvolts, nI, label=f'RF {rfv}V')
        ax.plot(energylist, nI, label=f'RF {rfv}V')
        # collectedx.append(defvolts)
        collectedx.append(energylist)
        collectedy.append(nI)
        ax.set(xlabel='Particle Energy (eV)',
               ylabel='~ nIons')
        # plt.xlim(1400, 2600)
        plt.ylim(-10, 300)
        plt.xlim(5500, 9900)
        plt.grid(b=True, which="major")
        plt.savefig(
            f'{deffolderout}{rfv}V.png',
            dpi=500)
        plt.close()
    # plottin comboplot
    fig, ax = plt.subplots()
    for k in range(collectedx.__len__()):
        ax.plot(collectedx[k], collectedy[k],
                label=f'RF {rfvlots[k]}V', linewidth=0.5)
    plt.grid(b=True, which="major")
    plt.legend()
    plt.ylim(-10, 300)
    plt.xlim(5500, 9900)
    ax.set(xlabel='Particle Energy (eV)',
           ylabel='arbitrary',
           title='deflector scan 2020-03-11 \n'
                 'conversion from deflector voltage to particle energy by simulations')
    plt.savefig(
        f'{deffolderout}combined.png',
        dpi=500)
    plt.close()

    # plotting comboplot but to investigate the tiny bumps
    fig, ax = plt.subplots()
    for k in range(collectedx.__len__()):
        ax.plot(collectedx[k], collectedy[k],
                label=f'RF {rfvlots[k]}V', linewidth=0.5)
    plt.grid(b=True, which="major")
    plt.legend()
    plt.xlim(7600, 10000)
    plt.ylim(-4, 25)
    ax.set(xlabel='Particle Energy (eV)',
           ylabel='arbitrary',
           title='deflector scan 2020-03-11 \n'
                 'focus on right bumps')
    plt.savefig(
        f'{deffolderout}combined_detail.png',
        dpi=500)
    plt.close()


rowfileplot()


def singlefileplot(name, ch, note):
    t = []
    ms = []
    results = []
    with open(f"{deffolderinp}{name}.csv") as csvfile:
        reader = csv.reader(
            csvfile)
        for row in reader:  # each row is a list
            results.append(row)
        # select specific time
        for i in results[2:]:
            t.append(float(i[0]))
            ms.append(float(i[ch]))
    fig, ax = plt.subplots()
    ax.plot(np.array(t) * 1e6, np.array(ms) * 1e3,
            linewidth=0.1)
    # plt.ylim(-50, 50)
    ax.set(xlabel='time (us)', ylabel=f'mV')
    plt.title(f'{note}')
    # plt.yticks(np.arange(-65, 65, 5))
    plt.grid(b=True)
    plt.savefig(
        f'{deffolderout}{name}.png',
        dpi=300)
    plt.close()
