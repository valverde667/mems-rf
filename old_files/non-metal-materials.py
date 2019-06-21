from warp import *

w3d.solvergeom = w3d.XZgeom

top.pline2   = 'Check ESQ potential'
top.pline1   = 'for different materials vs metal'
top.runmaker = 'Arun Persaud'

# --- Invoke setup routine (THIS IS MANDATORY).
setup()

# --- Set input parameters describing the 3d simulation.
# 1 mesh unit will be 2um, the structure will then consist of
# 250um vacuum, 2um metal, 500 um si, 2um box, 20 um si, 2um metal, 250um vacuum

gap_len = 250*um
metal_layer = 2*um
wafer = 500 *um
box = 2*um
SOI = 20*um

ysize = 1*cm
xsize = 100*um

radius = 90*um
spacer = 10*um

voltage = 100.

w3d.nx = 100
w3d.nz = int((gap_len + metal_layer + wafer + box + SOI + metal_layer + gap_len)/(1*um))

# --- Set to finite beam.
w3d.xmmin = -radius - spacer
w3d.xmmax =  radius + spacer
w3d.zmmin = 0.
w3d.zmmax = gap_len + metal_layer + wafer + box + SOI + gap_len

w3d.boundxy = 1

solver = MultiGrid2DDielectric()
registersolver(solver)

zpos = gap_len
box1t = Box(xsize, ysize, metal_layer, xcent=radius+xsize/2, zcent=zpos+metal_layer/2., voltage=+voltage)
box1b = Box(xsize, ysize, metal_layer, xcent=-radius-xsize/2, zcent=zpos+metal_layer/2., voltage=-voltage)
zpos += metal_layer
box2t = Box(xsize, ysize, wafer, xcent=radius+xsize/2, zcent=zpos+wafer/2., voltage=+voltage)
box2b = Box(xsize, ysize, wafer, xcent=-radius-xsize/2, zcent=zpos+wafer/2., voltage=-voltage)
zpos += wafer
boxstart = zpos
zpos += box
boxend = zpos
box4t = Box(xsize, ysize, SOI, xcent=radius+xsize/2, zcent=zpos+SOI/2., voltage=+voltage)
box4b = Box(xsize, ysize, SOI, xcent=-radius-xsize/2, zcent=zpos+SOI/2., voltage=-voltage)
zpos += SOI
box5t = Box(xsize, ysize, metal_layer, xcent=radius+xsize/2, zcent=zpos+metal_layer/2., voltage=+voltage)
box5b = Box(xsize, ysize, metal_layer, xcent=-radius-xsize/2, zcent=zpos+metal_layer/2., voltage=-voltage)
zpos += metal_layer

installconductor(box1t+box2t+box4t+box5t+box1b+box2b+box4b+box5b)

zboxstart = int(boxstart/(1*um))
zboxend = int(boxend/(1*um))
xstart1 = int(spacer/(2*radius+2*spacer)*100.)
xstart2 = 100-xstart1
solver.epsilon[:xstart1,boxstart:boxend] *= 3.9
solver.epsilon[xstart2:,boxstart:boxend] *= 3.9

# --- Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.).
package("w3d")
generate()

winon()

def doit():
    fieldsolve()
    fma()
    pfzx(filled=1)

