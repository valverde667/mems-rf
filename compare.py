#!/usr/bin/env python3
"""
Usage:
    compare.py <oldfile> <newfile>

"""

import numpy as np
from matplotlib import pyplot as plt
from docopt import docopt
import os
import sys

command = docopt(__doc__, version=1.0)

oldfile = command['<oldfile>']
newfile = command['<newfile>']
files = [oldfile, newfile]

# make sure files exists
for f in files:
    if not os.path.isfile(f):
        print("file: ", f, " does not exist")
        sys.exit()

olddata = np.load(oldfile)
newdata = np.load(newfile)

xa = olddata[0]
xb = newdata[0]
for ya, yb, in zip(olddata[1:], newdata[1:]):
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    ax[0].plot(xa, ya, label=oldfile)
    ax[0].plot(xb, yb, label=newfile)
    ax[0].set_xlabel("Time [s]")
    ax[0].legend(loc='best')

    ax[1].plot(xa, yb-ya)
    ax[1].set_xlabel("Time [s]")
    plt.title("diff: new-old")
    plt.suptitle("Comparing {} and {}".format(oldfile, newfile))
    plt.show()
