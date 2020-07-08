import numpy as np
import os, json, datetime, time, math, importlib, sys, scipy
from scipy.constants import elementary_charge
import matplotlib.pyplot as plt

pathtoparticlefiles = (
    "/home/timo/Documents/LBL/Warp/atap-meqalac-simulations/Energy Analyzer/v2_beams/"
)


def savejson(data, name):
    fp = f"{pathtoparticlefiles}{name}.json"
    with open(fp, "w") as writefile:
        json.dump(data, writefile, sort_keys=True, indent=1)


def readjson(name):
    fp = f"{name}"
    with open(fp, "r") as readfile:
        data = json.load(readfile)
    return data


# def histograms(prefilter,postfilter):
#     fig, (ax1, ax2) = plt.subplots(2)
#     ax1.hist(prefilter["ekinZ"], bins=range(4000,110000,1000))
#     ax2.hist(postfilter["ekinZ"], bins=range(4000,110000,500))
#     plt.savefig("histograms.png")
#     plt.savefig("histograms.svg")
def histograms(prefilter, postfilter):
    fig, (ax1, ax2) = plt.subplots(2)
    # ax1.hist(prefilter["ekinZ"], bins=range(4000,110000,1000))
    ax2.hist(postfilter["ekinZ"], range=(2, 10000))  # , bins=range(4000,110000,500))
    plt.savefig("histograms.png")
    plt.savefig("histograms.svg")
