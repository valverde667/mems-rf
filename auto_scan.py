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



