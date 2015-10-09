import subprocess
import tempfile
import os
import shutil

with open("esq.py", "r") as f:
    FILE = f.readlines()


for S in ['Argon', 'Xenon']:
    for E in [20e3, 40e3]:
        subprocess.call("git checkout master", shell=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.call("git checkout -b scan_{}_{}-finer".format(S, E),
                        shell=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        FILE[36] = "ekininit = {}\n".format(E)
        FILE[38] = "ions = Species(type={}, charge_state=1, name='{}')\n".format(S, S[:2])

        if E <30e3:
            start, stop, step = 200, 300, 10
        else:
            start, stop, step = 450, 600, 25
        for V in range(start, stop, step):
            FILE[27] = "Vesq = {:f}\n".format(float(V))

            with open("esq.py", "w+") as f:
                for line in FILE:
                    f.write(line)

            subprocess.call('git commit -am "{} {} {}"'.format(S, E, V), shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)


#if not FILE[26].startswith("gap"):
#    print(FILE[25:28])
#    print("no fit")
#    raise SystemExit
#
#subprocess.call("git checkout master", shell=True,
#                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#subprocess.call("git checkout -b scan_gap",
#                shell=True,
#                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#
#for V in range(100, 500, 50):
#    FILE[26] = "gap = {:f}*um\n".format(float(V))
#
#    with open("esq.py", "w+") as f:
#        for line in FILE:
#            f.write(line)
#
#    subprocess.call('git commit -am "gap {} um"'.format(V), shell=True,
#                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#
subprocess.call("git checkout master", shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

