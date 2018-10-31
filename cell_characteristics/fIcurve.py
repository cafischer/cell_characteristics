from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from cell_characteristics.analyze_APs import get_AP_onset_idxs


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
