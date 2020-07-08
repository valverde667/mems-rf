import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


p1 = np.random.rand(5,3)
p2 = np.random.rand(5,3)
p3 = np.random.rand(5,3)

xaverages = []
yaverages = []
zaverages = []

for rowindex in range(len(p1)):
    x1 = p1[rowindex,0]
    x2 = p2[rowindex,0]
    x3 = p3[rowindex,0]

    y1 = p1[rowindex,1]
    y2 = p2[rowindex,1]
    y3 = p3[rowindex,1]

    z1 = p1[rowindex,2]
    z2 = p2[rowindex,2]
    z3 = p3[rowindex,2]

    xmean = (x1+x2+x3)/3
    ymean = (y1+y2+y3)/3
    zmean = (z1+z2+z3)/3


    xaverages.append(xmean)
    yaverages.append(ymean)
    zaverages.append(zmean)
