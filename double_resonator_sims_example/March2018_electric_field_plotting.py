winon()
pfzr()
fma()
winon(1)
pfzx()
fma()
winon(2)
pfzr()
fma()
winon(3)
pfzx(fill=1, filled=1, plotselfe=2, comp='z', titles=0,cmin=0, cmax=1e5)
ptitles("Electric field", "Z [m]", "X [m]", "")
fma()
winon(4)
pfzx(fill=1, filled=1, plotselfe=2, comp='z', titles=0,cmin=0, cmax=1e5)
fma()
winon(5)
pfzr()
fma()
winon(6)
pfzx(comp='z')
fma()
winon(7)
plg(solver.getey()[solver.ix_axis,:], solver.zmesh)
fma()
winon(8)
plg(solver.getez()[solver.ix_axis,:], solver.zmesh)
fma()
winon(9)
plg(solver.getey()[solver.ix_axis,:], solver.zmesh)
fma()
winon(10)
plg(solver.getez()[solver.ix_axis,:], solver.zmesh)
# various tries to plot the field at the end of a Warp run
# march 2018
# figured out more neatly later with Grace Woods
#
fma()
solver.ix_axis
a=solver.getey()
a.size
a.shape
solver.zmesh
a
a.max()
a=solver.getex()
a.max()
b=solver.getez()
b.max()
solver.xmesh
winon(11)
plg(solver.getez()[16,:], solver.zmesh)
fma()
import matplotlib.pyplot as plt
a
plt.imshow(a)
plt.show()
help(plt.imshow)
winon(12)
plp(solver.getez()[solver.ix_axis,:], solver.zmesh)
fma()
winon(13)
plp(solver.getez()[solver.ix_axis,:], solver.zmesh)
fma()
plt.imshow(b,origin='lower')
plt.show()
plt.imshow(b,origin='lower')
plt.colorbar()
plt.show()
plt.imshow(a,origin='lower')
plt.colorbar()
plt.show()
winon(13)
winon(14)
pfzx()
fma()
winon(15)
plg(solver.getez()[16,:], solver.zmesh,label=True)
plg(solver.getez()[16,:], solver.zmesh,labels=True)