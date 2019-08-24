#x!/usr/local/bin/python3
"""
    Usage:
    auto_scan_try_2 [options] <parameter> <start> <end> <N>
    
    Options:
    
    """

import numpy as np
import os
import math
import datetime
import matplotlib.pyplot as plt
import sys
from docopt import docopt #to use commands via commandline
import array as arr
from pathlib import Path

import glob

commands = docopt(__doc__)
print(commands)

parameter_name = commands['<parameter>']
start = commands['<start>']
end = commands['<end>']
iterations = commands['<N>']

parameter_list = np.arange(float(start), float(end), float(iterations)) #this is working! just stops before last entered number in end

#print(parameter_list) #check if list has proper numbers

for i, x in enumerate(parameter_list):
    print(f"........This is the start of parameter change {i+1} for {parameter_name}........")
    os.system(f"python3 single-species-simulation.py --{parameter_name}={x}e-9")

print("so far so good")


"""for i, x in enumerate(


#make all of the parameters that we will use to scan

if parameter_name = 'bunch_length':
elif parameter_name = 'bunch_length':


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
"""
