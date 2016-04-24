"""Run a simulation and optimize results."""

import datetime
import json
import subprocess
import time
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

gaps = [0.00085634121477,
        0.00085834121477,
        0.00108734121477,
        0.00101154121477,
        0.00141094121477]

Vesq = [548*1.02**(i+1) for i in range(5)]
toffsets = [0.0e-9, 5.5e-9, 0.5e-9, 4.5e-9, 8.5e-9]

print("start")
vold = Vesq[0]
zoffset = 0
ttotal = 0.0
for i, [g, v, t] in enumerate(zip(gaps, Vesq, toffsets)):
    print("{}: gap={} V={}".format(i, g, v))
    tstart = time.time()
    command = "python3 unit-cell.py --cell={} --rfgap={} --Vesq={} --Vesqold={} --toffset={} --zoffset={} --t {}".format(
        i, g, v, vold, t, zoffset, ttotal)
    print("running: ", command)
    ret = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True)
    tstop = time.time()
    with open("mylog{:03d}.txt".format(i), "bw") as f:
        f.write(ret.stdout)
    with open("results{:03d}.json".format(i), "r") as f:
        results = json.load(f)
        zoffset = results['zoffsetout']
        ttotal = results['time']
        vold = v
    print("new zoffset: ", zoffset)
    print("done with one run. Time: {}".format(tstop-tstart))
print("done with simulations. start plotting")

results = []
results.append(np.load("../esqhist.npy"))
for i, g in enumerate(gaps):
    results.append(np.load("hist{:03d}.npy".format(i)))

# number of plots
N = results[0].shape[0]

with PdfPages('result.pdf') as pdf:
    for i in range(1, N):
        plt.figure()
        for r in results:
            plt.plot(r[0, :], r[i, :])
#        plt.plot(results[0][i, :])
#        Xstart = 0
#        for r in results[1:]:
#            Y = r[i, :]
#            X = np.arange(len(Y))+Xstart
#            Xstart = X[-1]
#            plt.plot(X, Y)
        pdf.savefig()
        plt.close()

    d = pdf.infodict()
    d['Title'] = 'Simulation test'
    d['Author'] = 'Arun Persaud'
    d['Subject'] = 'Testing simulation using unit cells'
    d['Keywords'] = 'warp, meqalac, optimization'
    d['CreationDate'] = datetime.datetime.now()
    d['ModDate'] = datetime.datetime.today()
