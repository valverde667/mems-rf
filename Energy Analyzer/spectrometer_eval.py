import json
import matplotlib.pyplot as plt

def scintillatorEval(self,inputpaths=["/media/timo/storage/Documents/LBL/Warp/atap-meqalac-simulations/Spectrometer-Sim/step2"]):
    print(f"combining {inputpaths.__len__()} files")
    # loading the files
    x=vx=y=vy=vz=ekin=[]
    for inp in inputpaths:
        with open(inp,"r") as fp:
            data = json.load(fp)
        x.append(data['x'])
        vx.append(data['vx'])
        y.append(data['y'])
        vy.append(data['vy'])
        vz.append(data['vz'])
        ekin.append(data['ekin'])


    x = zc.getx()
    vx = zc.getvx()
    y = zc.gety()
    vy = zc.gety()
    z = z_scintillator
    vz = zc.getvz()

    # using gist
    wp.plg(eKin(vz), x)
    wp.fma()
    # external plots
    # plt.plot(x,y,'ro')
    # plt.savefig(f'{thisFilesPath}/testing/{date}-xy.png')
    # plt.plot(x,vz,'ro')
    # plt.savefig(f'{thisFilesPath}/testing/{date}-xvz.png')
    plt.plot(x, eKin(vz), 'ro')
    plt.savefig(
        f'{thisFilesPath}/testing/{date}-ekinZx.png')
    plt.close()
    # Drawing energy Distribution
    ekinbin = np.arange(18, 27.5, 0.25)
    # ekin on hitting the screen
    # plt.hist(eKin2(vz,vx),bins=ekinbin)
    # plt.hist(eKin(vz),bins=ekinbin)
    # plt.savefig(f'{thisFilesPath}/testing/{date}-hist_ekin_scint.png')
    # plt.close()
    # ekin befor entering the spectrometer
    plt.hist(eKin(zc_test.getvz()), bins=ekinbin)
    plt.savefig(
        f'{thisFilesPath}/testing/{date}-hist_ekin_start.png')
    plt.close()
    # histogram of the amount of particles where on the screen
    plt.hist(x, bins=np.arange(0.018, 0.030, 0.00025))
    plt.savefig(
        f'{thisFilesPath}/testing/{date}-hist_xScint.png')
    plt.close()