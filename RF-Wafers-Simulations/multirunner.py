import os
import numpy as np
import threading, time

# this is hacky
pathtosssft = "/home/timo/Documents/Warp/atap-meqalac-simulations/sss-multi/"
subfolder = "testscan"
# commands = [
#     f'python3 {pathtosssft}single_species_simulation_for_thread.py '
#     f'--numRF 2 --name "2RF"',
#     f'python3 {pathtosssft}single_species_simulation_for_thread.py '
#     f'--numRF 4 --name "4RF"',
#     f'python3 {pathtosssft}single_species_simulation_for_thread.py '
#     f'--numRF 6 --name "6RF"'
# ]
commands = list()
# for i in range(1000,11000,1000):
#     commands.append(
#         f'python3 {pathtosssft}single_species_simulation_for_thread.py '
#         f'--numRF 2 --esq_voltage {i} --name "{i}Volt-RF2"'
#     )
# for i in range(1000,11000,1000):
#     commands.append(
#         f'python3 {pathtosssft}single_species_simulation_for_thread.py '
#         f'--numRF 4 --esq_voltage {i} --name "{i}Volt-RF4"'
#     )
# for i in range(1000,11000,1000):
#     commands.append(
#         f'python3 {pathtosssft}single_species_simulation_for_thread.py '
#         f'--numRF 6 --esq_voltage {i} --name "{i}Volt-RF6"'
#     )
# for i in range(1000,11000,1000):
#     commands.append(
#         f'python3 {pathtosssft}single_species_simulation_for_thread.py '
#         f'--numRF 8 --esq_voltage {i} --name "{i}Volt-RF8"'
#     )
for i in np.arange(0.6, 1.1, 0.05):
    commands.append(
        f"python3 {pathtosssft}single_species_simulation_for_thread.py "
        f'--numRF 10 --fraction {i} --name "'
        f'{i:,2f}fraction_RF10"'
    )

maxCores = 8 + 1
th = list()


def runcomm(com, x):
    os.system(com)


for i, c in enumerate(commands):
    while threading.active_count() >= maxCores:
        time.sleep(3)
        print(f"Active Threads :" f" {threading.active_count()}")
    print(f"starting command {i}")
    th.append(threading.Thread(target=runcomm, args=(c, 1)))
    th[i].start()
    print(f"started command {i}")

while threading.active_count() > 1:
    print(f"Finishing, Active Threads :" f" {threading.active_count()}")
    time.sleep(3)
print("DONE")
