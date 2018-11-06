from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from cell_characteristics.analyze_APs import get_AP_onset_idxs, get_AP_max_idx
from cell_characteristics import to_idx


def compute_fIcurve(v_traces, t_trace, amps, start_step, end_step):
    dur_step = end_step - start_step

    firing_rates = np.zeros(len(amps))
    for i, amp in enumerate(amps):
        AP_onsets = get_AP_onset_idxs(v_traces[i], threshold=0)
        n_APs = len(AP_onsets)
        firing_rates[i] = n_APs / dur_step * 1000  # convert to Hz
    return firing_rates


def compute_fIcurve_last_ISI(v_traces, t_trace, amps, start_step, end_step):
    dt = t_trace[1] - t_trace[0]

    firing_rates = np.zeros(len(amps))
    for i, amp in enumerate(amps):
        AP_onsets = get_AP_onset_idxs(v_traces[i], threshold=0)
        if len(AP_onsets) >= 2:
            firing_rates[i] = AP_onsets[-1] * dt - AP_onsets[-2] * dt
        else:
            firing_rates[i] = np.nan
    return firing_rates


def get_latency_to_first_spike(v, t, AP_onsets, start_step, end_step):
    dt = t[1] - t[0]
    if len(AP_onsets) >= 1 and start_step < t[AP_onsets[0]] < end_step:
        AP_max_idx = get_AP_max_idx(v, AP_onsets[0], AP_onsets[0] + to_idx(2, dt))
        latency = t[AP_max_idx] - start_step

        # print 'latency: ', latency
        # pl.figure()
        # pl.plot(t, v)
        # pl.plot(t[AP_max_idx], v[AP_max_idx], 'or')
        # pl.show()
        return latency
    else:
        return None


def get_ISI12(v, t, AP_onsets, start_step, end_step):
    dt = t[1] - t[0]
    if len(AP_onsets) >= 4 and start_step < t[AP_onsets[0]] and t[AP_onsets[-1]] < end_step:  # >=4 in case just doublet and one other spike this would be a bad estimate
        AP_max_idx1 = get_AP_max_idx(v, AP_onsets[0], AP_onsets[1])
        AP_max_idx2 = get_AP_max_idx(v, AP_onsets[1], AP_onsets[2])
        AP_max_idx3 = get_AP_max_idx(v, AP_onsets[2], AP_onsets[3])
        ISI1 = (AP_max_idx2 - AP_max_idx1) * dt
        ISI2 = (AP_max_idx3 - AP_max_idx2) * dt
        ISI12 = ISI1 / ISI2

        # print 'ISI12: ', ISI12
        # pl.figure()
        # pl.plot(t, v)
        # pl.plot(t[AP_max_idx1], v[AP_max_idx1], 'or')
        # pl.plot(t[AP_max_idx2], v[AP_max_idx2], 'or')
        # pl.plot(t[AP_max_idx3], v[AP_max_idx3], 'or')
        # pl.show()
        return ISI12
    else:
        return None