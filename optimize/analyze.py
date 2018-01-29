import glob
import json
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
from matplotlib.widgets import RadioButtons, Button
from matplotlib.text import Text
from matplotlib import cm
import numpy as np

files = glob.glob("res*json")

all = []
cells = []

for f in files:
    cell = int(f[8:10])
    cells.append(cell)
    runid = int(f[-9:-5])
    with open(f, "r") as input:
        data = json.load(input)

    data.update({'runid': runid})
    all.append(data)

cells = list(set(cells))

todo = ['toffset', 'zoffset', 'Ekin', 'VZ.std', 'Vesq',
        'X.std', 'XP.std', 'Y.std', 'YP.std', 'Z.std', 'rfgap']

fig = plt.figure()
ax = fig.gca(projection='3d')

X = RadioButtons(plt.axes([0.02, 0.0, 0.1, 0.2]), todo, active=0)
Y = RadioButtons(plt.axes([0.88, 0.0, 0.1, 0.2]), todo, active=1)
Z = RadioButtons(plt.axes([0.88, 0.5, 0.1, 0.2]), todo+['nvalue'], active=2)
C = RadioButtons(plt.axes([0.02, 0.8, 0.1, 0.2]), cells, active=0)

Go = Button(plt.axes([0.9, 0.9, 0.08, 0.08]), "plot")

ax.scatter([0, 1, 2, 3], [3, 2, 1, 1], [2, 2, 5, 5])

plotX = ""
plotY = ""
plotZ = ""
activecell = 0


def Xshow(label):
    global plotX
    plotX = label


def Yshow(label):
    global plotY
    plotY = label


def Zshow(label):
    global plotZ
    plotZ = label
    doplot("dummy")


def Cellshow(label):
    global activecell
    activecell = int(label)
    doplot("dummy")

Zbunch = (np.sqrt(2*(40e3)/131/(1.6e-27)*(1.6e-19)) *
          1e-9)  # length of a 1ns long beam at the source
Zrms = np.sqrt(1/12*Zbunch**2)  # sigma of a uniform distribution


def doplot(label):
    ax.cla()
    x = []
    y = []
    z = []
    for d in all:
        if d['cell'] == activecell:
            x.append(d[plotX])
            y.append(d[plotY])
            if plotZ == 'nvalue':
                nEkin = d['Ekin']
                nZrms = d['Z.std']
                nXrms = d['X.std']
                nYrms = d['Y.std']
                nXPrms = d['XP.std']
                nYPrms = d['YP.std']
                nvalue = [1/nEkin, 100*(nZrms-Zrms)**2, 1e6*(nXrms-20e-6)**2,
                          1e6*(nYrms-20e-6)**2, 0.2*(nXPrms-8e-3)**2, 0.2*(nYPrms-8e-3)**2]
                z.append(sum(nvalue))
            else:
                z.append(d[plotZ])
    ax.scatter(x, y, z)
    xi = np.sort(list(set(x)))
    yi = np.sort(list(set(y)))
    mX, mY = np.meshgrid(xi, yi)
    try:
        mZ = griddata(np.array([x, y]).T, np.array(z), (mX, mY), method='cubic')
        ax.plot_surface(mX, mY, mZ, cmap=cm.hot, alpha=0.2)
#        tupleList = zip(x, y, z)
#        poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
#        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='w', linewidth=1, alpha=0.5))
    except:
        pass

    ax.set(xlabel=plotX, ylabel=plotY, zlabel=plotZ)
    fig.canvas.draw()

X.on_clicked(Xshow)
Y.on_clicked(Yshow)
Z.on_clicked(Zshow)
C.on_clicked(Cellshow)
Go.on_clicked(doplot)

plt.show()
