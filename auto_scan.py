#x!/usr/local/bin/python3
"""
    Usage:
    auto_scan [options] <parameter> <start> <end> <N>
    
    Options:
    
    <parameter>
        bunch_length
        numRF
        rf_voltage
        esq_voltage
        fraction
        species_mass
        ekininit
        freq
        emit
        diva
    
    """

import numpy as np
import os
import math
from docopt import docopt #to use commands via commandline
from pathlib import Path

commands = docopt(__doc__)
print(commands)

parameter_name = commands['<parameter>']
start = commands['<start>']
end = commands['<end>']
iterations = commands['<N>']

"""if not parameter_name == 'bunch_length' or 'numRF' or 'rf_voltage' or 'esq_voltage'  or    'fraction' or 'species_mass' or 'ekininit' or 'freq' or 'emit' or 'diva':
    print("Please enter a valid parameter")
    exit()
else:"""

parameter_list = np.arange(float(start), float(end), float(iterations)) #stops before last entered number in end

    #print(parameter_list) #check if list has proper numbers

for i, x in enumerate(parameter_list):
    print(f"........This is the start of parameter change {i+1} for {parameter_name}........")
    os.system(f"python3 single-species-simulation.py --{parameter_name}={x}e-9")
