# Git notes: added condid :
#   change for be a module
import time
starttime = time.time()

import warpoptions

warpoptions.parser.add_argument('--energy', dest='energy',
                                type=float, default='10000')
warpoptions.parser.add_argument('--capV', dest='capV',
                                type=float, default='8000')

warpoptions.parser.add_argument('--dt', dest='timesteps',
                                type=float, default='5e-9')
            # 1e-9 is better but 5 is ok

import warp as wp
from warp import mm
import numpy as np
import os, json, datetime, time, math, importlib
# from geometry import spectrometer_standalone
from warp.particles.extpart import ZCrossingParticles


class spectrometerSimCalibration():
    def __init__(self, energy=10000, voltage=8000):
        self.particleenergy, self.voltage = energy, voltage
        importlib.reload(wp)

    RED = '\u001b[31m'
    RES = '\u001b[0m'

    def progressbar(self, f):
        f = f % 1
        f = int(f * 40)
        l = f * "="
        r = (40 - f) * "_"
        print(f"\u001b[31m progress : <<{l}{r}>> \u001b[0m")

    #
    #############################
    # moved from geometry file
    #############################
    def spectrometer_standalone(self, aperature=0.5 * mm,
                                centerpositionZ=39.2 * mm,
                                # b
                                distanceplates=(25 + 1.67) *
                                               mm):  # d
        voltage = self.voltage
        # Dimensions of metal plates
        plateX = 1.676 * mm
        plateY = 50 * mm
        plateZ = 25.3 * mm  # c
        # x-shift
        x2 = -(4.75 + 1.67) * mm  # e
        x1 = x2 + distanceplates
        plate1 = wp.Box(xsize=plateX, ysize=plateY,
                        zsize=plateZ, xcent=x1
                        , ycent=0, zcent=centerpositionZ,
                        voltage=-voltage / 2, condid=20)
        plate2 = wp.Box(xsize=plateX, ysize=plateY,
                        zsize=plateZ, xcent=x2
                        , ycent=0, zcent=centerpositionZ,
                        voltage=voltage / 2, condid=21)
        # aperture
        #   Dimensions
        apX = 12.5 * mm
        apY = 50 * mm
        apZ = 0.1 * mm
        #   calc center locations
        apXcenter = aperature / 2 + (apX / 2)
        apYcenter = 0 * mm
        apZcenter = 9.95 * mm  # a
        # setup
        aperture1 = wp.Box(xsize=apX, ysize=apY, zsize=apZ,
                           xcent=apXcenter, ycent=apYcenter,
                           zcent=apZcenter, voltage=0,
                           condid=10)
        aperture2 = wp.Box(xsize=apX, ysize=apY, zsize=apZ,
                           xcent=-apXcenter,
                           ycent=apYcenter,
                           zcent=apZcenter, voltage=0,
                           condid=11)

        return aperture1 + aperture2 + plate1 + plate2

    def eKin(self, arr):
        return np.square(
            arr) * .5 * 40 * wp.amu / wp.echarge / 1000

    def eKin2(self, arr1, arr2):
        'returns an array in keV'
        arrsq = np.add(np.square(arr1), np.square(arr2))
        return arrsq * .5 * 40 * wp.amu / wp.echarge / 1000

    #############################
    def simulate(self):
        print("starting simulation")
        wp.w3d.solvergeom = wp.w3d.XYZgeom
        wp.top.dt = warpoptions.options.timesteps
        # ToDo add warpoptions and add species to json
        selectedIons = wp.Species(
            charge_state=1, name='Ar', mass=40 * wp.amu)

        now = datetime.datetime.now()
        datetimestamp = \
            datetime.datetime.now().strftime(
                '%m-%d-%y_%H:%M:%S')
        date = datetime.datetime.now().strftime('%m-%d-%y')

        cgm_name = f'/home/timo/Documents/Warp/atap' \
                   f'-meqalac-simulations/Spectrometer' \
                   f'-Sim/step2/multirun_calibrations_output/' \
                   f'{self.particleenergy}eV_' \
                   f'{self.voltage}V_calibration' \
                   #f'{warpoptions.options.timesteps}'
        wp.setup(prefix=f'{cgm_name}')

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
        wp.w3d.zmmin = -.015  # dependent on
        # plateposition
        wp.w3d.zmmax = 0.12

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
        solver.mgtol = .5  # 1.0  # Poisson solver tolerance,
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
        print('setup geom')
        z_scintillator = 0.11112  # zsc
        zc_test = ZCrossingParticles(zz=0.01, laccumulate=1)
        zc = ZCrossingParticles(zz=z_scintillator,
                                laccumulate=1)
        conductors = self.spectrometer_standalone()
        # wp.installconductors(conductors)
        # print(f"Installed conductors : {wp.listofallconductors}")
        # recalculate fields

        # ParticleScraper # Todo use conducterID for statistics
        scraper = wp.ParticleScraper(conductors,
                                     lsavecondid=True)
        # Setup Scintillator Screen
        scintillatorScreen = wp.ZPlane(
            zcent=z_scintillator + .0001,
            voltage=0, condid=30)
        wp.installconductors(conductors)
        wp.installconductors(scintillatorScreen)
        sciScraper = wp.ParticleScraper(scintillatorScreen,
                                        lsaveintercept=True)
        wp.fieldsol(-1)
        #############################
        #   add one Particle
        #############################
        print('add Particles')
        vz = math.sqrt(
            2 * self.particleenergy * wp.echarge / (
                    40 * wp.amu))
        selectedIons.addparticles(x=[0], y=[0], z=[0],
                                  vx=[0], vy=[0],
                                  vz=[vz],
                                  lallindomain=True)
        # lallindomain in particles/particles.py line 1798
        #############################
        #   plot of E-fields:
        #############################
        print('efield plots')
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

        print('run Sim')
        time_time = []
        tmax = 5e-7  # rough calculation

        while (selectedIons.getn() != 0):
            #self.progressbar(wp.top.time / tmax)
            wp.step(10)
            time_time.append(wp.top.time)
            #   ToDo setup voltage range properly
            #     wp.pfzx(fill=1, filled=1, plotselfe=True, comp='z',
            #             titles=0, cmin=-1.2 * 1000,
            #             cmax=1.2 * 1000)

            # color sort by time of birth or energy
            # wp.ptitles("Particles and Fields", "Z [m]", "X [m]")
            # wp.fma()
            wp.pfzx(fill=1, filled=1, plotphi=1, comp='E',
                    plotselfe=True, iy=15)
            selectedIons.color = 'red'
            selectedIons.ppzx()  # shows how the beam changes shape
            wp.fma()  # second frame in cgm file
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
            'energy': self.particleenergy,
            'x': x,
            'vx': vx,
            'y': y,
            'vy': vy,
            'z': z,
            'vz': vz,
            'ekinz': ekinz,
            'voltage': self.voltage,
            'timesteps': warpoptions.options.timesteps,
            'runtime': time.time() - starttime,
            'angle': np.arctan(vx[0] / vz[0]) /(2*np.pi)
                     *360,
            'ekinxz':1/2*40 * wp.amu *(
                    vx[0]*vx[0]+vz[0]*vz[0])/wp.echarge,

        }
        op = f'{cgm_name}.json'
        with open(op, 'w') as fp:
            json.dump(data, fp)


if __name__ == '__main__':
    print(f'starting Simulation with E')
    s = spectrometerSimCalibration(
        warpoptions.options.energy,
        warpoptions.options.capV)
    s.simulate()
    print('DONE')
