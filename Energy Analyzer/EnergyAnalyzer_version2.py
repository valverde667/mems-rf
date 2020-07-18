"""
Example Pierce diode with subsequest solenoid transport.
Hot plate source emitting singly ionized potassium.
"""
from warp import *
from warp.particles.extpart import ZCrossingParticles
import numpy as np
import json
from EnergyAnalyzer_version2_utils import *

top.pline1 = "Energy analyzer Version 2"
top.runmaker = "Timo Bauer"

# --- Invoke setup routine for the plotting
setup()

# --- Set the dimensionality
w3d.solvergeom = w3d.XYZgeom

# --- Basic parameters
beam_radius = 0.5 * mm

# --- Setup simulation species
beam = Species(type=Argon, charge_state=+1, name="beam", mass=40 * amu)

# --- Set basic beam parameters
beam.a0 = beam_radius
beam.b0 = beam_radius
beam.ap0 = 0.0e0
beam.bp0 = 0.0e0
beam.ibeam = 1e-6
# beam.vthz     = sqrt(source_temperature*jperev/beam.mass)
# beam.vthperp  = sqrt(source_temperature*jperev/beam.mass)
# derivqty()

# --- Variables to set symmetry, when using 3D
# w3d.l4symtry = true
# w3d.l2symtry = false

# --- Set boundary conditions
# ---   for field solve
w3d.bound0 = dirichlet
w3d.boundnz = neumann
w3d.boundxy = neumann
# ---   for particles
top.pbound0 = absorb
top.pboundnz = absorb
# top.prwall = channel_radius

# --- Set field grid size
w3d.xmmin = -15e-3
w3d.xmmax = 30e-3
w3d.ymmin = -1e-3
w3d.ymmax = 1e-3
w3d.zmmin = -15e-3
w3d.zmmax = 140e-3

# w3d.xmmin = -15e-3
# w3d.xmmax = 100e-3

# --- Field grid dimensions - note that nx and ny must be even.
w3d.nx = 300
w3d.ny = 5
w3d.nz = 500

# --- Set the time step size. This needs to be small enough to satisfy the Courant limit.
top.dt = 1e-9  # 7e-10

# --- Specify injection of the particles
top.inject = 1  # 2 means space-charge limited injection
# top.rinject = source_curvature_radius # Source radius of curvature
top.npinject = 0#5  # Approximate number of particles injected each step
top.vinject = 5
top.ekin = 5
# w3d.l_inj_exact = true


f3d.mgtol = 1.0e-0  # Multigrid solver convergence tolerance, in volts

solver = MRBlock3D()
registersolver(solver)

### FLAGS
calibration = False
manysingleparticles = True

### Set up geometry

deflector = slit_before_plates() + deflectionPlates(5500)
installconductor(deflector)
scraper_deflector = ParticleScraper(deflector)

if calibration:
    filter = tiltedInfiniteBox(z_front=0.1, x_front=3e-3, thickness=5e-3, angle=4.28400412 * deg, voltage=0)
    installconductor(filter)
    scraper_filter = ParticleScraper(filter, lsavecondid=True, lsaveintercept=True)
else:
    slitpos = list(zip([0.09912104,0.09956524,0.09934374,0.09978553], [0.01246544,0.00653563,0.00949254,0.00359491]))
    filter = slittedTiltedBox(slitpos, 4.28400412 * deg, 1e-3, 25e-3, 60e-3, 1, 0)
    installconductor(filter)
    scraper_filter = ParticleScraper(filter, lsavecondid=True, lsaveintercept=True)
    #
    target = tiltedInfiniteBox(z_front=0.1+0.025+0.01, x_front=3e-3, thickness=5e-3, angle=4.28400412 * deg, voltage=0)
    installconductor(target)
    scraper_target = ParticleScraper(target, lsavecondid=True, lsaveintercept=True)

# top.pline1 = ("Injected beam. Semi-Gaus. %dx%d. npinject=%d, dt=%d"%(w3d.nx, w3d.nz, top.npinject, top.dt))

package("w3d")
generate()

# --- Open up plotting windows
winon(0)
winon(1)


@callfromafterstep
def beamplots(final=False):
    # print(beam.gety())
    if top.it % 4 == 0:
        window(0)
        fma()
        pfzx(
            fill=1,
            filled=1,
            plotphi=1,
            comp="E",
            plotselfe=1,
            inverted=1,
            cond=1,
            titles=False,
        )
        beam.ppzx(titles=False)
        #deflector.draw(filled=100, color="red")
        #filter.draw(filled=150, color="red")
        ptitles("Example")
        refresh()
    #

    if top.it == 3:
        window(1)
        pfzx(
            fill=1,
            filled=1,
            plotphi=1,
            comp="E",  # "x",
            plotselfe=1,
            inverted=1,
            cond=1,
            titles=False,
        )
        deflector.draw(filled=150, color="red")
        filter.draw(filled=1, color="red")
        refresh()

    if top.it % 2 == 0:
        window(1)
        beam.ppzx(titles=False)
        refresh()

    if final:
        window(1)
        beam.ppzx(titles=False)
        refresh()
        fma()
        print("Saving plots")
        window(1)
        beam.ppzx(titles=False)
        refresh()
        fma()


def examplebeamplots():
    window(0)
    fma()
    pfzr(plotsg=0, cond=0, titles=False)
    deflector.draw(filled=150, fullplane=False)
    # plate.draw(filled=100, fullplane=False)
    # pipe.draw(filled=100, fullplane=False)
    # plotsolenoids()
    ppzr(titles=False)
    # limits(w3d.zmminglobal, w3d.zmmaxglobal, 0., channel_radius)
    ptitles("Hot plate source into solenoid transport", "Z (m)", "R (m)")
    refresh()

    window(1)
    fma()
    pzcurr()
    # limits(w3d.zmminglobal, w3d.zmmaxglobal, 0., diode_current*1.5)
    refresh()


def nonlinearsource():
    NP = int(time_prof(top.time))
    x = random.normal(bunch_centroid_position[0], bunch_rms_size[0], NP)
    y = random.normal(bunch_centroid_position[1], bunch_rms_size[1], NP)
    z = bunch_centroid_position[2]
    vx = random.normal(bunch_centroid_velocity[0], bunch_rms_velocity[0],
                       NP)
    vy = random.normal(bunch_centroid_velocity[1], bunch_rms_velocity[1],
                       NP)
    vz = picmi.warp.clight * np.sqrt(1 - 1.0 / (beam_gamma ** 2))
    beam.wspecies.addparticles(
        x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, gi=1.0 / beam_gamma, w=bunch_w
    )


# def singleParticleInjection(energy, n_beams=3):


def singleParticleInjection(energy=100000, n_beams=3):
    top.inject = 0

    def velocity(ekin):
        return np.sqrt(2 * ekin * echarge / (40 * amu))

    beam.addparticles(
        x=list(np.linspace(0, (n_beams - 1) * 3e-3, n_beams)),
        y=[0] * n_beams,
        z=[0] * n_beams,
        vx=[0] * n_beams,
        vy=[0] * n_beams,
        vz=[velocity(energy)] * n_beams,
        lallindomain=True,
    )

def particleDistributionInjection(n_beams=4):
    amp = 1e-9
    ppers = amp / echarge
    pperstep = ppers * top.dt
    print(f"Particles per timestep  : {pperstep}")

    def constant(min, max):
        return np.random.random() * (max - min) + min

    def gaussian(max):
        return np.random.standard_normal() * max

    #
    top.inject = 0

    for i in range(int(pperstep)):
        energy = constant(5000, 110000)
        beam.addparticles(
            x=list(
                np.linspace(0, (n_beams - 1) * 3e-3, n_beams)
                + constant(-beam_radius, beam_radius)
            ),
            y=np.array([0]) * n_beams + constant(-beam_radius,
                                                 beam_radius),
            z=[0] * n_beams,
            vx=[0] * n_beams,
            vy=[0] * n_beams,
            vz=[velocity(energy)] * n_beams,
            lallindomain=True,
        )


def zcrossing_save(zc, name):
    if zc.getn():
        zc_data = {
            "ekinZMinMaxAvg": [
                energy(min(zc.getvz())),
                energy(max(zc.getvz())),
                energy(mean(zc.getvz())),
            ],
            "x": zc.getx().tolist(),
            "y": zc.gety().tolist(),
            "z": zc.getzz(),
            "vx": zc.getvx().tolist(),
            "vy": zc.getvy().tolist(),
            "vz": zc.getvz().tolist(),
            "angle": np.tan(zc.getvx() / zc.getvz()).tolist(),
            "ekinZ": energy(zc.getvz()).tolist(),
            "t": zc.gett().tolist(),
        }
        with open(name, "w") as writefile:
            json.dump(zc_data, writefile, sort_keys=False, indent=1)
        return zc_data


if calibration:
    singleParticleInjection(energy=100000, n_beams=4)
    while beam.getn() or top.it == 0:
        step(1)
    # lost COND ID: top.pidlost[:, -1]
    lostangles = arctan(top.uxplost / top.uzplost)
    print(f"Angles : {lostangles[:4]}\n"
          f"Angles deg : {lostangles[:4] / deg}\n"
          f"Energies : {energy(top.uzplost)[:4]}\n"
          f"z pos : {top.zplost[:4]}\n"
          f"x pos : {top.xplost[:4]}\n")
if manysingleparticles:
    step(1)
    for e in np.arange(120e3,40000,-1000):
        singleParticleInjection(energy=e,n_beams=4)
        step(5)
    while beam.getn():
        step(1)
    uz, ux, uy, zz, xx, yy = lostByConductor()
    print(f"PASSED Particles : {len(uz)}")
    print(f"PASSED ENERGY : {energy(np.array(uz))}")

else:
    singleParticleInjection(energy=100000, n_beams=4)
    while beam.getn() or top.it == 0:
        step(1)
    # lost COND ID: top.pidlost[:, -1]
    lostangles = arctan(top.uxplost / top.uzplost)
    print(f"Angles : {lostangles[:4]}\n"
          f"Angles deg : {lostangles[:4] / deg}\n"
          f"Energies : {energy(top.uzplost)[:4]}\n"
          f"z pos : {top.zplost[:4]}\n"
          f"x pos : {top.xplost[:4]}\n")
beamplots(True)
time.sleep(5)

# # ZCrossings
# zc1 = ZCrossingParticles(zz=0.1, laccumulate=1)
# zc_injection = ZCrossingParticles(zz=2e-3, laccumulate=1)
# zc_postfilter = ZCrossingParticles(zz=0.19, laccumulate=1)
#
# # Injection
# # singleParticleInjection()
# # particleDistributionInjection()
#
# while beam.getn() or top.it == 0:
#     if top.it < 600:
#         particleDistributionInjection()
#     print(f"{beam.getn()}")
#     print(f"Max Energy : {energy(max(beam.getvz()))}")
#     print(f"Min Energy : {energy(min(beam.getvz()))}")
#     step(1)
# beamplots(True)
#
# zcrossing_save(zc1, "zc1.json")
# predata = zcrossing_save(zc_injection, "zc_injection.json")
# postdata = zcrossing_save(zc_postfilter, "zc_postfilter.json")
#
# histograms(predata, postdata)
# print(
#     f'Current Ratio Pre and post filter: {len(postdata["x"]) / len(predata["x"])}')

# for i in range(50000,0,-10000):
#     step(100)
#     top.ekin=i


####
# pfzx(fill=1, filled=1, plotphi=1, comp="x", plotselfe=1,
#         inverted=0, cond=1)
# for i in range(100):
#     step(1)
#     print(f"Particles {beam.getn()}")
#     # beam.color = "black"  # "red"
#     # beam.ppzx()
# fma()
####


# --- Make sure that last plot frames get sent to the cgm file
# window(0)
# hcp()
# window(1)
# hcp()


#######################

# def spectrometer_standalone(voltage=500):
#     n_beams = 3
#     # Deflector Plates
#     centerpositionZ = 53 * mm  # b
#     distanceplates = 25 * mm  # d
#     # Dimensions of metal plates
#     plateX = 1 * mm
#     plateY = 50 * mm
#     plateZ = 50.8 * mm  # c
#     # x-shift
#     x2 = -6 * mm  # e
#     x1 = x2 + distanceplates
#     plate1 = Box(
#         xsize=plateX,
#         ysize=plateY,
#         zsize=plateZ,
#         xcent=x1,
#         ycent=0,
#         zcent=centerpositionZ,
#         voltage=-voltage / 2,
#         condid=20,
#     )
#     plate2 = Box(
#         xsize=plateX,
#         ysize=plateY,
#         zsize=plateZ,
#         xcent=x2,
#         ycent=0,
#         zcent=centerpositionZ,
#         voltage=voltage / 2,
#         condid=21,
#     )
#     # aperture
#     # build a box and punch holes in it
#     d_aperture = 1 * mm
#     # setup
#     aperture = Box(
#         xsize=40 * mm,
#         ysize=40 * mm,
#         zsize=0.1 * mm,
#         xcent=7 * mm,
#         ycent=0,
#         zcent=0,
#         voltage=0,
#         condid=10,
#     )
#     holes = [
#         ZCylinder(
#             zcent=0,
#             ycent=0,
#             xcent=i * 3 * mm,
#             radius=d_aperture,
#             length=0.1 * mm,
#             voltage=0,
#         )
#         for i in range(n_beams)
#     ]
#     slit_before_plates = aperture - sum(holes)
#     # filters:
#     # build a box and punch holes in it
#     n_filters = 3
#     phis_filter = [0.07526778307059949, 0.07438315246073729,
#                    0.07400417250218974]
#     pos_filter_z = [0.1, 0.14, 0.18]  # g
#     phi_offsets = [
#         (p - pos_filter_z[0]) * np.tan(ph) for p, ph in
#         zip(pos_filter_z, phis_filter)
#     ]
#     thickness_filterplate = 0.5e-3
#     filter_apertures = [1e-3, 1e-3, 1e-3]  # diameter
#     bias_filters = [10, 20, 30]  # voltage on the plates
#     initial_heights = [
#         0.003621128630106844,
#         0.0065801180436158855,
#         0.009557295653103096,
#     ]  # f
#     #
#     print(f"Phi_offset : \n{phi_offsets}")
#     #
#     filters = []
#     for apt, pos, vol, phi_offset in zip(
#             filter_apertures, pos_filter_z, bias_filters, phi_offsets
#     ):
#         plate = Box(
#             xsize=100 * mm,
#             ysize=50 * mm,
#             zsize=thickness_filterplate,
#             xcent=initial_heights[0],
#             ycent=0,
#             zcent=pos,
#             voltage=vol,
#         )
#         holes = [
#             ZCylinder(
#                 zcent=pos,
#                 ycent=0,
#                 xcent=initial_height + phi_offset,
#                 radius=apt / 2,
#                 length=thickness_filterplate,
#                 voltage=0,
#             )
#             for initial_height in initial_heights
#         ]
#         filters.append(plate - sum(holes))
#     return plate1 + plate2 + slit_before_plates, sum(filters)
