import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from heka_reader import HekaReader, get_indices_for_protocol

def convert_unit_prefix(from_prefix, x):
    """
    Converts x from unit prefix to base unit.
    :param from_prefix: Prefix (implemented are 'da', 'd', 'c', 'm', 'u', 'n').
    :type from_prefix:str
    :param x: Quantity to convert.
    :type x: array_like
    :return: Converted quantity.
    :rtype: array_like
    """
    if from_prefix == 'T':
        return x * 1e12
    elif from_prefix == 'M':
        return x * 1e6
    elif from_prefix == 'h':
        return x * 1e2
    elif from_prefix == 'da':
        return x * 1e1
    elif from_prefix == 'd':
        return x * 1e-1
    elif from_prefix == 'c':
        return x * 1e-2
    elif from_prefix == 'm':
        return x * 1e-3
    elif from_prefix == 'u':
        return x * 1e-6
    elif from_prefix == 'n':
        return x * 1e-9
    elif from_prefix == 'p':
        return x * 1e-12
    else:
        raise ValueError('No valid prefix!')


def get_cellarea(L, diam):
    """
    Takes length and diameter of some cell segment and returns the area of that segment (assuming it to be the surface
    of a cylinder without the circle surfaces as in Neuron).
    :param L: Length (um).
    :type L: float
    :param diam: Diameter (um).
    :type diam: float
    :return: Cell area (cm).
    :rtype: float
    """
    return L * diam * np.pi


if __name__ == '__main__':
    file_dir = '../data/2015_08_06d/2015_08_06d.dat'
    hekareader = HekaReader(file_dir)
    protocol = 'hypTester'
    indices = get_indices_for_protocol(hekareader, protocol)
    correct_ljp = True
    ljp = -0.016

    v_traces = np.zeros((len(indices), len(hekareader.get_xy(indices[0])[0])))
    for i, index in enumerate(indices):
        t, v = hekareader.get_xy(index)
        v_traces[i, :] = v
        if correct_ljp:
            v_traces[i, :] -= ljp
        t_unit, v_unit = hekareader.get_units_xy(index)

    # compute mean
    v_mean = np.mean(v_traces, 0)
    pl.figure()
    pl.plot(t, v_mean, 'k')
    pl.xlabel('Time (' + t_unit + ')')
    pl.ylabel('Membrane Potential (' + v_unit + ')')
    #pl.show()

    # load i_inj
    i_inj = np.array(pd.read_csv('../data/Protocols/'+protocol+'.csv', header=None)[0].values)
    start_i_inj = np.where(i_inj < 0)[0][0]
    end_i_inj = np.where(i_inj[start_i_inj:] == 0)[0][0] + start_i_inj

    # fit tau (decay time constant)
    def exponential(t, tau):
        return v_diff * np.exp(-t / tau) - v_diff

    max_depolarization_idx = np.argmin(v_mean)
    v_expdecay = v_mean[start_i_inj:max_depolarization_idx] - v_mean[start_i_inj]
    t_expdecay = t[start_i_inj:max_depolarization_idx] - t[start_i_inj]
    v_diff = np.abs(v_expdecay[-1] - v_expdecay[0])
    tau = curve_fit(exponential, t_expdecay, v_expdecay)[0][0]

    pl.figure()
    pl.plot(t_expdecay, v_expdecay, 'k')
    pl.plot(t_expdecay, exponential(t_expdecay, tau), 'r')
    #pl.show()

    # compute Rin
    v_rest = np.mean(v_mean[:start_i_inj - 1])
    v_in = np.min(v_mean) - v_rest
    i_in = i_inj[start_i_inj]

    r_in = np.abs(v_in / i_in) * 1000  # V / nA to MOhm

    # compute capacitance
    c_m = tau / r_in * 10 ** 6  # sec / MOhm to pF
    print 'V_rest: ' + str(v_rest*1000) + ' mV'
    print 'tau: ' + str(tau * 1000) + ' ms'
    print 'R_in: ' + str(r_in) + ' MOhm'
    print 'c_m: ' + str(c_m) + ' pF'

    # estimate cell size
    c_m_ind = 1.0 * 1e6  # pF/cm2
    cell_area = 1.0/(c_m_ind / c_m)  # cm2
    diam = np.sqrt(cell_area/np.pi) * 1e4  # um
    print 'Estimated diam: ' + str(diam)
    #
    L = 100  # um
    diam = 100  # um
    print 'If L={0} and diam={1}: '.format(L, diam)
    L = L * 1e-4  # cm
    diam = diam * 1e-4  # cm
    cell_area = get_cellarea(L, diam)  # cm2

    g_pas = 1 / convert_unit_prefix('M', r_in) / cell_area  # S/cm2
    print 'g_pas: ' + str(g_pas) + ' S/cm2'
    c_m_per_area = c_m * 1e-6 / cell_area  # uF/cm2
    print 'c_m: ' + str(c_m_per_area) + ' uF/cm2'