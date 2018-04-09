import numpy as np
import geometry as geo

timestep_size = 5e-11
ev_to_joules = 1.6022e-19
amu_to_kg = 1.66054e-27
gap_width = geo.RF_gap

def fraction_of_v_max_gained(e_kin_not, entry_coefficient, mass, v_max, freq):
    e_kin = e_kin_not
    poffset = np.arcsin(entry_coefficient)
    time = 0
    position = 0

    while position < gap_width:
        voltage = v_max*np.sin(2*np.pi*freq*time+poffset) # find votage at that timestep
        velocity = np.sqrt(2*e_kin*ev_to_joules/(mass*amu_to_kg)) # find velocity at that timestep
        if (velocity*timestep_size+position < gap_width):
            distance_gained = velocity*timestep_size # find distance traveled in that timestep
        else:
            distance_gained = gap_width-position
        e_kin += voltage*distance_gained/gap_width # add energy gained in that timestep to energy
        time += timestep_size # advance time
        position += distance_gained
    return (e_kin-e_kin_not)/v_max, time # fraction of max voltage gained

def optimize_entry_coefficient(e_kin_not, desired_gain, mass, v_max, freq):
    posibilities = np.linspace(0, 1, 10000)
    results = [abs(desired_gain - fraction_of_v_max_gained(e_kin_not, posibility, mass, v_max, freq)[0]) for posibility in posibilities]
    best = posibilities[results.index(np.min(results))]
    return best

def entry_coefficients(num_rf, e_kin_not, desired_gain, mass, v_max, freq):
    coefficients = []
    e_kin = e_kin_not
    for i in range(num_rf):
        coefficient = optimize_entry_coefficient(e_kin, desired_gain, mass, v_max, freq)
        e_kin += v_max*desired_gain
        coefficients.append(coefficient)
    return coefficients

def delta_ts(coefficients, e_kins, mass, v_max, freq):
    i = 0
    j = 1
    delta_ts = []
    delta_xs = []
    while i < len(coefficients)-1:
        delta_ts.append((np.arcsin(coefficients[j])-np.arcsin(coefficients[i])+np.pi)/(2*np.pi)*(1/freq)-fraction_of_v_max_gained(e_kins[i], coefficients[i], mass, v_max, freq)[1])
        delta_xs.append(delta_ts[len(delta_ts)-1]*np.sqrt(2*e_kins[j]*ev_to_joules/(mass*amu_to_kg)))
        i+=1
        j+=1
    return delta_ts, delta_xs
