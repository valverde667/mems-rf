#!/usr/bin/python3
"""Run a simulation and optimize results.

Usage: run.py [--startcell CELL]

Options:
  --startcell CELL    start at cell CELL, skiping the once before [default: 0]

"""

import datetime
import docopt
import glob
import json
import subprocess
import sys
import time
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool

import logging

commands = docopt.docopt(__doc__, version="0.1")
startcell = int(commands['--startcell'])

logging.basicConfig(filename='run.log', level=logging.DEBUG)
logging.info('-'*30)
logging.info('new run {}'.format(datetime.datetime.now().isoformat()))

gaps = [0.00085634121477,
        0.00085834121477,
        0.00108734121477,
        0.00101154121477,
        0.00141094121477]

Vesq = [548*1.02**(i+1) for i in range(5)]
toffsets = [0.0e-9, 5.5e-9, 0.5e-9, 4.5e-9, 8.5e-9]


def single_run(cell=0, runid=0, rfgap=1e-3, Vesq=500, Vesqold=500,
               toffset=0, zoffset=0, t=0):
    """runs a single simulation"""

    print("{}-{}: gap={} V={}".format(cell, runid, g, v))
    tstart = time.time()
    command = ["python3",
               "unit-cell.py",
               "--cell={}".format(cell),
               "--run={}".format(runid),
               "--rfgap={}".format(rfgap),
               "--Vesq={}".format(Vesq),
               "--Vesqold={}".format(Vesqold),
               "--toffset={}".format(toffset),
               "--zoffset={}".format(zoffset),
               "--t={}".format(t)]
    logging.info("{}-{} command: {}".format(cell, runid, " ".join(command)))
    ret = subprocess.run(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    tstop = time.time()
    print("done with {}-{}: {}".format(cell, runid, tstop-tstart))
    with open("mylog{:03d}-{:04d}.txt".format(i, runid), "bw") as f:
        f.write(ret.stdout)
    return runid, tstop-tstart

print("start")
vold = Vesq[0]
zoffset = 0
ttotal = 0.0
bestcells = []
v40kv = np.sqrt(2*(40e3)/131/(1.6e-27)*(1.6e-19))
freq = 100e6
gapstart = 0.0008
gapend = 0.0013
Vesqstart = 500
Vesqend = 570
for i, [g, v, t] in enumerate(zip(gaps, Vesq, toffsets)):
    if i < startcell:
        continue
    runid = 0
    print("starting pool")
    with Pool(7) as pool:
        res = []
        for tt in np.linspace(0, 10e-9, 8):
            for gg in np.linspace(gapstart, gapend, 3):
                for vv in np.linspace(Vesqstart, Vesqend, 3):
                    print("starting run {}-{}".format(i, runid))
                    logging.info("cell {} runid {} t {} g{} v{}".format(i, runid, gg, vv, tt))
                    res.append(pool.apply_async(single_run, (i, runid, gg, vv, vold,
                                                             tt, zoffset, ttotal)))
                    runid += 1
        results = [r.get(timeout=None) for r in res]
    print("finished pool for {} workers".format(len(res)))
    bestrun = None
    value = 1e8
    Zbunch = v40kv * 1e-9  # length of a 1ns long beam at the source
    Zrms = np.sqrt(1/12*Zbunch**2)  # sigma of a uniform distribution
    for r in results:
        id, t = r
        with open("results{:03d}-{:04d}.json".format(i, id), "r") as f:
            results = json.load(f)
            if 'Error' in results:
                print("{}-{}".format(i, id), results['Error'])
                continue
            nEkin = results['Ekin']
            nZrms = results['Z.std']
            nXrms = results['X.std']
            nYrms = results['Y.std']
            nXPrms = results['XP.std']
            nYPrms = results['YP.std']
            Warn = results['Warning']
            if Warn != "":
                print("{}-{} WARNINGS: {}".format(i, id, Warn))
        nvalue = [1/nEkin, 100*(nZrms-Zrms)**2, 1e6*(nXrms-20e-6)**2,
                  1e6*(nYrms-20e-6)**2, 0.2*(nXPrms-8e-3)**2, 0.2*(nYPrms-8e-3)**2]
        if sum(nvalue) < value:
            bestrun = id
            value = sum(nvalue)
    logging.info("cell {} best runid {}".format(i, bestrun))
    if bestrun is None:
        print("couldn't find an optimal value for cell", i)
        sys.exit(0)
    with open("results{:03d}-{:04d}.json".format(i, bestrun), "r") as f:
        results = json.load(f)
        zoffset = results['zoffsetout']
        ttotal = results['time']
        nEkin = results['Ekin']
        nVesq = results['Vesq']
        vold = v
    gapstart = np.sqrt(2*(nEkin-5e3)/131/(1.6e-27)*(1.6e-19))
    gapend = np.sqrt(2*(nEkin+15e3)/131/(1.6e-27)*(1.6e-19))
    Vesqstart = nVesq*0.97
    Vesqend = Vesqstart*1.05
    # copy the particles from the best run in to a file for the next run
    subprocess.run(['cp', 'save-{:03d}-{:04d}.pkl'.format(i, bestrun), 'save-{:03d}.pkl'.format(i)])
    bestcells.append(bestrun)
print("done with simulations. start plotting")

results = []
results.append(np.load("../esqhist.npy"))
for i, id in enumerate(bestcells):
    results.append(np.load("hist{:03d}-{:04d}.npy".format(i, id)))

# number of plots
N = results[0].shape[0]

with PdfPages('result.pdf') as pdf:
    for i in range(1, N):
        plt.figure()
        for r in results:
            plt.plot(r[0, :], r[i, :])
        pdf.savefig()
        plt.close()

    d = pdf.infodict()
    d['Title'] = 'Simulation test'
    d['Author'] = 'Arun Persaud'
    d['Subject'] = 'Testing simulation using unit cells'
    d['Keywords'] = 'warp, meqalac, optimization'
    d['CreationDate'] = datetime.datetime.now()
    d['ModDate'] = datetime.datetime.today()
