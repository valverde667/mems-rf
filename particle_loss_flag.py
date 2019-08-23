#program to sort through resulting particle loss txt files to find which files have particle loss and what change caused the particle loss
#maybe we should also include a graph showing the final resulting particle loss (last(array)/max(array) with corresponding titles and axis

"""
    Usage:
    particle_loss_flag.py [options] <parameter>"""


import numpy as np
import os
import math
import datetime
import matplotlib.pyplot as plt
from docopt import docopt #to use commands via commandline

import glob

commands = docopt(__doc__)
print(commands)

parameter_name = commands['<parameter>']

#bunch_length = np.arange(5, 200, 5)

#make a flag to keep track of which files have particle loss could possibly use the last amount of particles recorded and compare it to the maximum value in the funciton
surviving_particles = []
fraction_particles_kept = []

#I must only read in the files that are more recent with headings that are on a seperate line
# so far I think I can only use this in "batches" of parameter scans. I need to be able to write and read in files that keep track of the parameters used. Maybe I could write the text files differently? Include what stayed the same and what was changed in the first two lines with the number that was used for the change. This way you can create points for each text file and graph them against what was changed



txt_files = sorted(glob.glob(f"{parameter_name}*.txt"))

for i in txt_files:
    with open(f"{i}", "r") as file:
        parameter = f.readline()
        changed_parameter = float(parameter.split("=")[1].strip())
        header = f.readline()
        data = f.readline()
        #particles = float(data.split(",")[1].strip())
        surviving_particles.append(data)
    
    #print(header)
    
if np.max(surviving_particles) > surviving_particles[-1]:
        fraction_particles_kept = surviving_particles[-1]/np.max(surviving_particles)
        surviving_particles.append(fraction_particles_kept)

#make a graph of the surviving particles against each parameter change
plt.scatter(bunch_length, surviving_particles)
plt.xlabel("Bunch_Length")
plt.ylabel("Fraction of Surviving Particles")
plt.title("Surviving Particles vs Bunch Length")

fig.savefig("5-200e-9_bunch_length_vs_surving_particles.pdf")
