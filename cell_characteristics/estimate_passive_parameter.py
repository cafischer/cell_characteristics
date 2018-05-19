import matplotlib.pyplot as pl
import numpy as np
from scipy.optimize import curve_fit
from cell_characteristics import to_idx
from scipy.signal import argrelmin
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function, shift_v_rest
pl.style.use('paper')


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
    :param L: Length.
    :type L: float
    :param diam: Diameter.
    :type diam: float
    :return: Cell area.
    :rtype: float
    """
    return L * diam * np.pi


def estimate_passive_parameter(v, t, i_inj):
    """
    Protocol: negative step current.
    Assumes cm = 1 uF/cm**2
    :param v: Membrane potential.
    :param t: Time.
    :param i_inj: Injected current.
    :return: c_m, r_in, tau_m, cell_area, diam, g_pas: Capacitance (pF), input resistance (MOhm),
    membrane time constant (ms), cell area (cm**2), diameter (um), passive/leak conductance (S/cm**2)
    """

    start_step = np.where(np.diff(np.abs(i_inj)) > 0)[0][0] + 1
    end_step = np.where(-np.diff(np.abs(i_inj)) > 0)[0][0]

    # fit tau
    peak_hyperpolarization = argrelmin(v[start_step:end_step], order=to_idx(5, t[1]-t[0]))[0][0] + start_step
    v_expdecay = v[start_step:peak_hyperpolarization] - v[start_step]
    t_expdecay = t[start_step:peak_hyperpolarization] - t[start_step]
    v_diff = np.abs(v_expdecay[-1] - v_expdecay[0])

    def exp_decay(t, tau):
        return v_diff * np.exp(-t / tau) - v_diff

    tau_m, _ = curve_fit(exp_decay, t_expdecay, v_expdecay)  # ms
    tau_m = tau_m[0]

    pl.figure()
    pl.plot(t, v, 'k')
    pl.plot(t_expdecay+start_step*(t[1]-t[0]), exp_decay(t_expdecay, tau_m) + v[0], 'r', label='fitted exp. decay',
            linewidth=1.5)
    pl.legend()
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    pl.tight_layout()
    pl.show()

    # compute Rin
    last_fourth_i_inj = 3/4 * (end_step - start_step) + start_step

    v_rest = np.mean(v[0:start_step - 1])
    i_rest = np.mean(i_inj[0:start_step - 1])
    v_in = np.mean(v[last_fourth_i_inj:end_step]) - v_rest
    i_in = np.mean(i_inj[last_fourth_i_inj:end_step]) - i_rest

    r_in = v_in / i_in  # mV / nA = MOhm

    # compute capacitance
    c_m = tau_m / r_in * 1000  # ms / MOhm to pF

    # estimate cell size
    c_m_ind = 1.0 * 1e6  # pF/cm2  # from experiments
    cell_area = 1.0 / (c_m_ind / c_m)  # cm2
    diam = np.sqrt(cell_area / np.pi) * 1e4  # um

    # estimate g_pas
    g_pas = 1 / convert_unit_prefix('M', r_in) / cell_area  # S/cm2

    print 'tau: ' + str(tau_m) + ' ms'
    print 'Rin: ' + str(r_in) + ' MOhm'
    print 'c_m: ' + str(c_m) + ' pF'
    print 'cell_area: ' + str(cell_area) + ' cm2'
    print 'diam: ' + str(diam) + ' um'
    print 'g_pas: ' + str(g_pas) + ' S/cm2'

    return c_m, r_in, tau_m, cell_area, diam, g_pas


if __name__ == '__main__':
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data/2015_08_20e.dat'
    protocol = 'hypTester'
    v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(data_dir, protocol, group='Group1', trace='Trace1',
                                                     sweep_idxs=None, return_sweep_idxs=True)
    i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, t_mat[0, -1],  t_mat[0, 1] - t_mat[0, 0])
    ljp = -16
    v_mat = shift_v_rest(v_mat, ljp)

    # compute mean
    v_mean = np.mean(v_mat, 0)
    # pl.figure()
    # pl.plot(t_mat[0, :], v_mean, 'k')
    # pl.xlabel('Time (ms)')
    # pl.ylabel('Membrane Potential (mV)')
    # #pl.show()
    #
    # pl.figure()
    # for v in v_mat:
    #     pl.plot(t_mat[0, :], v)
    # pl.xlabel('Time (ms)')
    # pl.ylabel('Membrane Potential (mV)')
    # pl.show()

    # estimate passive parameters
    c_m, r_in, tau_m, cell_area, diam, g_pas = estimate_passive_parameter(v_mean, t_mat[0, :], i_inj_mat[0, :])