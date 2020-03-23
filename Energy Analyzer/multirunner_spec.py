import os
import threading, time
from spectrometer_sim import spectrometerSim

# this is hacky but works
pathtoinput = '/home/timo/Documents/Warp/atap-meqalac-simulations/Spectrometer-Sim/step1/'
pathtoinput += "testscan"

maxCores = 8 +1


def startSim(pfile, pout, fout, v):
    s = spectrometerSim(pfile, pout, fout, v)
    s.simulate()


tr = list()
for counter in range(1000, 10000, 500):
    tr.append(threading.Thread(target=startSim, args=(
        f"{pathtoinput}/10000Volt_snap_0.00mm.json",
        "/home/timo/Documents/Warp/atap-meqalac"
        "-simulations/Spectrometer-Sim/step2/",
        f"capvolt_{counter}",
        counter
    )))
    tr[-1].start()
    while threading.active_count() >= maxCores:
        print(f'Active Threads :'
              f' {threading.active_count()}')
        time.sleep(3)

while threading.active_count():
    print(f'Finishing, Active Threads :'
          f' {threading.active_count()}')
    time.sleep(3)
print("DONE")