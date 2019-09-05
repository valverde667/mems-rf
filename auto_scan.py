#x!/usr/local/bin/python3
"""
    Usage:
    auto_scan [options] <parameter> <start> <end> <power> <N>
    
    Options:
    
    <parameter>
        bunch_length 1e-9
        numRF 4
        rf_voltage 5000-10000 ~8000
        esq_voltage OFF=.01 ON=100-850
        fraction .8-1 ~.8
        species_mass 40
        ekininit 10e3
        freq 13.56e3-27e3
        emit .25e-3-.5e-3 ~.37e-3
        diva 5e-3-29e-3
    
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
power = commands['<power>']
iterations = commands['<N>']

"""if not parameter_name == 'bunch_length' or 'numRF' or 'rf_voltage' or 'esq_voltage'  or    'fraction' or 'species_mass' or 'ekininit' or 'freq' or 'emit' or 'diva':
    print("Please enter a valid parameter")
    exit()
else:"""

parameter_list = np.arange(float(start), float(end), float(iterations)) #stops before last entered number in end

    #print(parameter_list) #check if list has proper numbers

for i, x in enumerate(parameter_list):
    print(f"........This is the start of parameter change {i+1} for {parameter_name}........")
    os.system(f"python3 single-species-simulation.py --{parameter_name}={x}e{power}")
