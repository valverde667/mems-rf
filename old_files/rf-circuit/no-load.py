import ahkab
from ahkab import circuit, printing, time_functions
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import numpy as np


def tune(C, load=3e-12, start=8e6, stop=20e6):
    c = circuit.Circuit(title="STS-50 rf-circuit")

    gnd = c.get_ground_node()

    c.add_capacitor('c1', n1='n1', n2='n10', value=C)
    # to damp things a bit and get rounded resonances in the first plots
    c.add_resistor('rint', n1='n10', n2='n2', value=0.11)
    c.add_capacitor('c2', n1='n3', n2=gnd, value=load)
    c.add_resistor('r1', n1='n3', n2=gnd, value=100e6)
    c.add_inductor('l1', n1='n1', n2='n2', value=1.21e-6)
    c.add_inductor('l2', n1='n2', n2='n3', value=0.77e-6)
    c.add_inductor_coupling('k1', 'l1', 'l2', 0.99)

    voltage = time_functions.sin(vo=0.0, va=3, freq=15e6)

    c.add_vsource('V1', n1='n1', n2=gnd, dc_value=0, ac_value=1000, function=voltage)

    op_analysis = ahkab.new_op()
    for i in range(5):
        ac_analysis = ahkab.new_ac(start=start, stop=stop, points=50, x0=None)

        r = ahkab.run(c, an_list=[ac_analysis])
        f = r['ac']['f']
        VV = np.abs(r['ac']['Vn3'])
        ff = f[np.argmax(VV)]
        myrange = (stop-start)/5
        start = ff-myrange/2
        stop = ff+myrange/2
    return VV.max()

fig, (ax1, ax2) = plt.subplots(2, 1)

cap = np.linspace(10e-12, 250e-12, 30)
freq = np.linspace(8e6, 20e6, 60)
df = np.diff(freq).mean()
V = []
for f in freq:
    print("working on f", f)
    tmp = []
    for c in cap:
        tmp.append(tune(c, load=3e-12, start=f-df/2, stop=f+df/2))
    V.append(tmp)
V = np.array(V)

f, c = np.meshgrid(freq*1e-6, cap*1e12)
ax1.pcolormesh(f, c, V.T)  # , norm=colors.LogNorm(vmin=c.min(), vmax=c.max()))
ax1.set_title("no load (3 pF)")
ax1.set_xlabel("Freq. [Hz]")
ax1.set_ylabel("tuning Cap. [pF]")

cap = np.linspace(10e-12, 250e-12, 30)
freq = np.linspace(8e6, 20e6, 30)

V = []
for f in freq:
    print("working on f", f)
    tmp = []
    for c in cap:
        tmp.append(tune(c, load=12e-12, start=f-df/2, stop=f+df/2))
    V.append(tmp)
V = np.array(V)

f, c = np.meshgrid(freq*1e-6, cap*1e12)
ax2.pcolormesh(f, c, V.T)  # , norm=colors.LogNorm(vmin=c.min(), vmax=c.max()))
ax2.set_title("with load (12 pF)")
ax2.set_xlabel("Freq. [Hz]")
ax2.set_ylabel("tuning Cap. [pF]")

# plt.xlabel('f [MHz]')
# plt.show()
plt.show()
