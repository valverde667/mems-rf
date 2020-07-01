# Git notes: added condid :
#   change for be a module
import warpoptions
import warp as wp
from warp import mm
import numpy as np
import os, json, datetime, time, math, importlib, sys

# from geometry import spectrometer_standalone
from warp.particles.extpart import ZCrossingParticles


class spectrometerSim:
    def __init__(
        self,
        pathtoparticlefiles="/home/timo/Documents/LBL/Warp/atap-meqalac-simulations/Energy Analyzer/v2_beams/",
        pathtooutput="/home/timo/Documents/LBL/Warp/atap-meqalac-simulations/Energy Analyzer/v2_out/",
        outputfile="default",
        voltage=4000,
    ):
        # init
        self.pathtoparticlefiles, self.pathtooutput, self.outputfilename = (
            pathtoparticlefiles,
            pathtooutput,
            outputfile,
        )
        self.voltage = voltage

    #############################
    # moved from geometry file
    #############################
    # CAREFUL WITH FILES!!!!
    def spectrometer_standalone(
        self,
        d_aperture=1 * mm,
        centerpositionZ=38.4 * mm,  # b
        distanceplates=23.324 * mm,  # d
        n_beams=3,
    ):
        voltage = self.voltage
        # Dimensions of metal plates
        plateX = 1 * mm
        plateY = 50 * mm
        plateZ = 50.8 * mm  # c
        # x-shift
        x2 = -3 * mm  # e
        x1 = x2 + distanceplates
        plate1 = wp.Box(
            xsize=plateX,
            ysize=plateY,
            zsize=plateZ,
            xcent=x1,
            ycent=0,
            zcent=centerpositionZ,
            voltage=-voltage / 2,
            condid=20,
        )
        plate2 = wp.Box(
            xsize=plateX,
            ysize=plateY,
            zsize=plateZ,
            xcent=x2,
            ycent=0,
            zcent=centerpositionZ,
            voltage=voltage / 2,
            condid=21,
        )
        # aperture
        # build a box and punch holes in it
        # setup
        aperture = wp.Box(
            xsize=20 * mm,
            ysize=50 * mm,
            zsize=5 * mm,
            xcent=0,
            ycent=0,
            zcent=0,
            voltage=0,
            condid=10,
        )
        holes = [
            wp.ZCylinder(
                zcent=0,
                ycent=0,
                xcent=i * 3 * mm,
                radius=d_aperture,
                length=5 * mm,
                voltage=0,
            )
            for i in range(n_beams)
        ]
        return aperture - wp.sum(holes) + plate1 + plate2

    def eKin(self, arr):
        return np.square(arr) * 0.5 * 40 * wp.amu / wp.echarge / 1000

    def eKin2(self, arr1, arr2):
        "returns an array in keV"
        arrsq = np.add(np.square(arr1), np.square(arr2))
        return arrsq * 0.5 * 40 * wp.amu / wp.echarge / 1000

    #############################
    def simulate(self):
        wp.w3d.solvergeom = wp.w3d.XYZgeom
        wp.top.dt = 1e-9
        # ToDo add warpoptions and add species to json
        selectedIons = wp.Species(charge_state=1, name="Ar", mass=40 * wp.amu)

        now = datetime.datetime.now()
        datetimestamp = datetime.datetime.now().strftime("%m-%d-%y_%H:%M:%S")
        date = datetime.datetime.now().strftime("%m-%d-%y")

        # with open(self.particlefile, "r") as fp:
        #     particledata = json.load(fp)

        # cgm setup
        cgm_name = f"{self.pathtooutput}/{self.outputfilename}"
        wp.setup(prefix=f"{cgm_name}")

        # Symmetries - there are none
        wp.w3d.l4symtry = False
        wp.w3d.l2symtry = False

        #############################
        # Set solver and grid sizes:
        #############################
        wp.w3d.xmmin = -0.03
        wp.w3d.xmmax = 0.1
        wp.w3d.ymmin = -0.002
        wp.w3d.ymmax = 0.002
        wp.w3d.zmmin = -0.015  # dependent on
        # plateposition
        wp.w3d.zmmax = 0.2

        # grids in window
        wp.w3d.nx = 100
        wp.w3d.ny = 30
        wp.w3d.nz = 200

        wp.top.nhist = 5
        #############################
        #   Adding stuff to fix stuff
        #############################
        # --- Set boundary conditions
        # ---   for field solve
        wp.w3d.bound0 = wp.dirichlet
        wp.w3d.boundnz = wp.neumann
        wp.w3d.boundxy = wp.neumann
        # ---   for particles
        # wp.top.pbound0 = wp.absorb
        # wp.top.pboundnz = wp.absorb
        # wp.top.prwall = wp.channel_radius
        #############################

        ############################# copy of sss
        # --- Save time histories of various quantities versus z.
        wp.top.lhpnumz = True
        wp.top.lhcurrz = True
        wp.top.lhrrmsz = True
        wp.top.lhxrmsz = True
        wp.top.lhyrmsz = True
        wp.top.lhepsnxz = True
        wp.top.lhepsnyz = True
        wp.top.lhvzrmsz = True

        # --- Set up fieldsolver
        solver = wp.MRBlock3D()
        wp.registersolver(solver)
        solver.mgtol = 0.5  # 1.0  # Poisson solver tolerance,
        # in volts
        solver.mgparam = 1.5
        solver.downpasses = 3  # 2
        solver.uppasses = 3  # 2
        solver.gridmode = 0

        # --- Generate the PIC code
        # (allocate storage, load ptcls, t=0 plots, etc.)
        wp.package("w3d")
        wp.generate()
        #############################

        #############################
        #   setup geometry
        #############################
        # Comment: ParticleScraper does not store the velocity of
        # a particle, but ZCrossingParticles does. Therefore
        print("setup geom")
        z_scintillator = 0.199
        zc_test = ZCrossingParticles(zz=0.01, laccumulate=1)
        zc = ZCrossingParticles(zz=z_scintillator, laccumulate=1)
        conductors = self.spectrometer_standalone()
        # wp.installconductors(conductors)
        # print(f"Installed conductors : {wp.listofallconductors}")
        # recalculate fields

        # ParticleScraper # Todo use conducterID for statistics
        scraper = wp.ParticleScraper(conductors, lsavecondid=True)
        # Setup Scintillator Screen
        scintillatorScreen = wp.ZPlane(
            zcent=z_scintillator + 0.0001, voltage=0, condid=30
        )
        wp.installconductors(conductors)
        wp.installconductors(scintillatorScreen)
        sciScraper = wp.ParticleScraper(scintillatorScreen, lsaveintercept=True)
        wp.fieldsol(-1)
        #############################
        #   add Particles
        #############################
        print("add Particles")
        self.particleenergy = 25000
        vz = math.sqrt(2 * self.particleenergy * wp.echarge / (40 * wp.amu))
        selectedIons.addparticles(
            x=[0], y=[0], z=[-10e-3], vx=[0], vy=[0], vz=[vz], lallindomain=True
        )
        # print("add Particles")
        # pd = particledata
        # # for x, y, z, vx, vy, vz in zip(pd['x'], pd['y'], pd['z'],
        # #                                pd['vx'], pd['vy'], pd['vz']):
        # #     # print(f"adding particle at {x}, {y}, {z}")
        # #     selectedIons.addparticles(x=x, y=y, z=z, vx=vx,
        # #                               vy=vy, vz=vz,
        # #                               lallindomain=True)
        # # can be an array
        # pdz = np.array(pd["z"])
        # z = pdz - pdz.mean()
        # print(z[3])
        # selectedIons.addparticles(
        #     x=pd["x"],
        #     y=pd["y"],
        #     z=z,
        #     vx=pd["vx"],
        #     vy=pd["vy"],
        #     vz=pd["vz"],
        #     lallindomain=True,
        # )
        # lallindomain in particles/particles.py line 1798
        #############################
        #   plot of E-fields:
        #############################
        print("efield plots")
        # # does not plot contours of potential
        # wp.pfzx(fill=1, filled=1, plotphi=0)
        # wp.fma()
        # plots contours of potential
        # wp.pfzx(fill=1, filled=1, plotphi=1)
        # wp.fma()
        # wp.fieldsol(-1)

        #############################
        #   running the simulation
        #############################

        print("run Sim")
        time_time = []
        tmax = 1e-8  # rough calculation

        while wp.top.time < tmax and selectedIons.getn() != 0:
            # self.progressbar(wp.top.time / tmax)
            wp.step(10)
            time_time.append(wp.top.time)
            #   ToDo setup voltage range properly
            #     wp.pfzx(fill=1, filled=1, plotselfe=True, comp='z',
            #             titles=0, cmin=-1.2 * 1000,
            #             cmax=1.2 * 1000)

            # color sort by time of birth or energy
            # wp.ptitles("Particles and Fields", "Z [m]", "X [m]")
            # wp.fma()
            wp.pfzx(fill=1, filled=1, plotphi=1, comp="E", plotselfe=True, iy=15)
            selectedIons.color = "red"
            selectedIons.ppzx()  # shows how the beam changes shape
            wp.fma()  # second frame in cgm file
            # avEkin = np.square(selectedIons.getvz()).mean() * \
            #         .5 * 40 * wp \
            #             .amu / wp.echarge / 1000
            # print(f"EKIN {avEkin}")
            print(f"# particles : {selectedIons.getn()}")
        #####################
        #   export
        #####################
        x = zc.getx().tolist()
        vx = zc.getvx().tolist()
        y = zc.gety().tolist()
        vy = zc.gety().tolist()
        z = z_scintillator
        vz = zc.getvz().tolist()
        ekinz = self.eKin(vz).tolist()
        data = {
            "x": x,
            "vx": vx,
            "y": y,
            "vy": vy,
            "z": z,
            "vz": vz,
            "ekinz": ekinz,
            "voltage": self.voltage,
        }
        if not os.path.isdir(self.pathtooutput):
            os.mkdir(self.pathtooutput)
        op = f"{self.pathtooutput}/{self.outputfilename}.json"
        with open(op, "w") as fp:
            json.dump(data, fp)

        # todo remove
        # avEkin = np.square(selectedIons.getvz()).mean() * \
        #          .5 * 40 * wp \
        #              .amu / wp.echarge / 1000
        # minEkin = min(selectedIons.getvz()) ** 2 * .5 * 40 \
        #           * wp.amu / wp.echarge / 1000
        # maxEkin = max(selectedIons.getvz()) ** 2 * .5 * 40 \
        #           * wp.amu / wp.echarge / 1000
        # print(f"EKIN min av max {minEkin} | {avEkin} | {maxEkin}")


###########################################################
###########################################################
###########################################################


# class spectrometerSimCalibration:
#     def __init__(self, energy=10000, voltage=8000):
#         self.particleenergy, self.voltage = energy, voltage
#         importlib.reload(wp)
#
#     RED = "\u001b[31m"
#     RES = "\u001b[0m"
#
#     def progressbar(self, f):
#         f = f % 1
#         f = int(f * 40)
#         l = f * "="
#         r = (40 - f) * "_"
#         print(f"\u001b[31m progress : <<{l}{r}>> \u001b[0m")
#
#     #
#     #############################
#     # moved from geometry file
#     #############################
#     def spectrometer_standalone(
#         self, aperature=0.5 * mm, centerpositionZ=40 * mm, distanceplates=20 * mm
#     ):
#         voltage = self.voltage
#         # Dimensions of metal plates
#         plateX = 1 * mm
#         plateY = 50 * mm
#         plateZ = 25.3 * mm
#         # x-shift
#         x2 = -3 * mm
#         x1 = x2 + distanceplates
#         plate1 = wp.Box(
#             xsize=plateX,
#             ysize=plateY,
#             zsize=plateZ,
#             xcent=x1,
#             ycent=0,
#             zcent=centerpositionZ,
#             voltage=-voltage / 2,
#             condid=20,
#         )
#         plate2 = wp.Box(
#             xsize=plateX,
#             ysize=plateY,
#             zsize=plateZ,
#             xcent=x2,
#             ycent=0,
#             zcent=centerpositionZ,
#             voltage=voltage / 2,
#             condid=21,
#         )
#         # aperture
#         #   Dimensions
#         apX = 10 * mm
#         apY = 50 * mm
#         apZ = 0.1 * mm
#         #   calc center locations
#         apXcenter = aperature / 2 + (apX / 2)
#         apYcenter = 0 * mm
#         apZcenter = 20 * mm
#         # setup
#         aperture1 = wp.Box(
#             xsize=apX,
#             ysize=apY,
#             zsize=apZ,
#             xcent=apXcenter,
#             ycent=apYcenter,
#             zcent=apZcenter,
#             voltage=0,
#             condid=10,
#         )
#         aperture2 = wp.Box(
#             xsize=apX,
#             ysize=apY,
#             zsize=apZ,
#             xcent=-apXcenter,
#             ycent=apYcenter,
#             zcent=apZcenter,
#             voltage=0,
#             condid=11,
#         )
#
#         return aperture1 + aperture2 + plate1 + plate2
#
#     def eKin(self, arr):
#         return np.square(arr) * 0.5 * 40 * wp.amu / wp.echarge / 1000
#
#     def eKin2(self, arr1, arr2):
#         "returns an array in keV"
#         arrsq = np.add(np.square(arr1), np.square(arr2))
#         return arrsq * 0.5 * 40 * wp.amu / wp.echarge / 1000
#
#     #############################
#     def simulate(self):
#         print("starting simulation")
#         wp.w3d.solvergeom = wp.w3d.XYZgeom
#         wp.top.dt = 1e-9
#         # ToDo add warpoptions and add species to json
#         selectedIons = wp.Species(charge_state=1, name="Ar", mass=40 * wp.amu)
#
#         now = datetime.datetime.now()
#         datetimestamp = datetime.datetime.now().strftime("%m-%d-%y_%H:%M:%S")
#         date = datetime.datetime.now().strftime("%m-%d-%y")
#
#         cgm_name = (
#             f"/home/timo/Documents/Warp/atap-meqalac-simulations/Spectrometer-Sim/step2/Calibrations/"
#             f"{self.particleenergy}eV_"
#             f"{self.voltage}V_calibration"
#         )
#         wp.setup(prefix=f"{cgm_name}")
#
#         # Symmetries - there are none
#         wp.w3d.l4symtry = False
#         wp.w3d.l2symtry = False
#
#         #############################
#         # Set solver and grid sizes:
#         #############################
#         wp.w3d.xmmin = -0.03
#         wp.w3d.xmmax = 0.1
#         wp.w3d.ymmin = -0.002
#         wp.w3d.ymmax = 0.002
#         wp.w3d.zmmin = -0.015  # dependent on
#         # plateposition
#         wp.w3d.zmmax = 0.12
#
#         # grids in window
#         wp.w3d.nx = 100
#         wp.w3d.ny = 30
#         wp.w3d.nz = 200
#
#         wp.top.nhist = 5
#         #############################
#         #   Adding stuff to fix stuff
#         #############################
#         # --- Set boundary conditions
#         # ---   for field solve
#         wp.w3d.bound0 = wp.dirichlet
#         wp.w3d.boundnz = wp.neumann
#         wp.w3d.boundxy = wp.neumann
#         # ---   for particles
#         # wp.top.pbound0 = wp.absorb
#         # wp.top.pboundnz = wp.absorb
#         # wp.top.prwall = wp.channel_radius
#         #############################
#
#         ############################# copy of sss
#         # --- Save time histories of various quantities versus z.
#         wp.top.lhpnumz = True
#         wp.top.lhcurrz = True
#         wp.top.lhrrmsz = True
#         wp.top.lhxrmsz = True
#         wp.top.lhyrmsz = True
#         wp.top.lhepsnxz = True
#         wp.top.lhepsnyz = True
#         wp.top.lhvzrmsz = True
#
#         # --- Set up fieldsolver
#         solver = wp.MRBlock3D()
#         wp.registersolver(solver)
#         solver.mgtol = 0.5  # 1.0  # Poisson solver tolerance,
#         # in volts
#         solver.mgparam = 1.5
#         solver.downpasses = 3  # 2
#         solver.uppasses = 3  # 2
#         solver.gridmode = 0
#
#         # --- Generate the PIC code
#         # (allocate storage, load ptcls, t=0 plots, etc.)
#         wp.package("w3d")
#         wp.generate()
#         #############################
#
#         #############################
#         #   setup geometry
#         #############################
#         # Comment: ParticleScraper does not store the velocity of
#         # a particle, but ZCrossingParticles does. Therefore
#         print("setup geom")
#         z_scintillator = 0.1
#         zc_test = ZCrossingParticles(zz=0.01, laccumulate=1)
#         zc = ZCrossingParticles(zz=z_scintillator, laccumulate=1)
#         conductors = self.spectrometer_standalone()
#         # wp.installconductors(conductors)
#         # print(f"Installed conductors : {wp.listofallconductors}")
#         # recalculate fields
#
#         # ParticleScraper # Todo use conducterID for statistics
#         scraper = wp.ParticleScraper(conductors, lsavecondid=True)
#         # Setup Scintillator Screen
#         scintillatorScreen = wp.ZPlane(
#             zcent=z_scintillator + 0.0001, voltage=0, condid=30
#         )
#         wp.installconductors(conductors)
#         wp.installconductors(scintillatorScreen)
#         sciScraper = wp.ParticleScraper(scintillatorScreen, lsaveintercept=True)
#         wp.fieldsol(-1)
#         #############################
#         #   add one Particle
#         #############################
#         print("add Particles")
#         vz = math.sqrt(2 * self.particleenergy * wp.echarge / (40 * wp.amu))
#         selectedIons.addparticles(
#             x=[0], y=[0], z=[0], vx=[0], vy=[0], vz=[vz], lallindomain=True
#         )
#         # lallindomain in particles/particles.py line 1798
#         #############################
#         #   plot of E-fields:
#         #############################
#         print("efield plots")
#         # # does not plot contours of potential
#         # wp.pfzx(fill=1, filled=1, plotphi=0)
#         # wp.fma()
#         # plots contours of potential
#         # wp.pfzx(fill=1, filled=1, plotphi=1)
#         # wp.fma()
#         # wp.fieldsol(-1)
#
#         #############################
#         #   running the simulation
#         #############################
#
#         print("run Sim")
#         time_time = []
#         tmax = 5e-7  # rough calculation
#
#         while selectedIons.getn() != 0:
#             self.progressbar(wp.top.time / tmax)
#             wp.step(10)
#             time_time.append(wp.top.time)
#             #   ToDo setup voltage range properly
#             #     wp.pfzx(fill=1, filled=1, plotselfe=True, comp='z',
#             #             titles=0, cmin=-1.2 * 1000,
#             #             cmax=1.2 * 1000)
#
#             # color sort by time of birth or energy
#             # wp.ptitles("Particles and Fields", "Z [m]", "X [m]")
#             # wp.fma()
#             wp.pfzx(fill=1, filled=1, plotphi=1, comp="E", plotselfe=True, iy=15)
#             selectedIons.color = "red"
#             selectedIons.ppzx()  # shows how the beam changes shape
#             wp.fma()  # second frame in cgm file
#             print(f"# particles : {selectedIons.getn()}")
#         #####################
#         #   export
#         #####################
#         x = zc.getx().tolist()
#         vx = zc.getvx().tolist()
#         y = zc.gety().tolist()
#         vy = zc.gety().tolist()
#         z = z_scintillator
#         vz = zc.getvz().tolist()
#         ekinz = self.eKin(vz).tolist()
#         data = {
#             "energy": self.particleenergy,
#             "x": x,
#             "vx": vx,
#             "y": y,
#             "vy": vy,
#             "z": z,
#             "vz": vz,
#             "ekinz": ekinz,
#             "voltage": self.voltage,
#         }
#         op = f"{cgm_name}.json"
#         with open(op, "w") as fp:
#             json.dump(data, fp)
