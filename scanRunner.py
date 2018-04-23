import subprocess
from multiprocessing import Pool
import time
from itertools import repeat
import numpy as np
from itertools import permutations

perm = permutations([11, 12, 13, 14], 2)
x = [[11,11],[12,12],[13,13],[14,14]]
for i in list(perm):
    x.append(i)

print(x)

def f(x):
    subprocess.run(["python","massSelector.py","--selected_mass", str(x[0]), "--mass", str(x[1])])

pool=Pool(processes=4)

pool.map(f,x)
