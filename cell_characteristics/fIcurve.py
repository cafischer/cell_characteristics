import numpy as np
from analyze_APs import get_AP_onset_idxs


def compute_fIcurve(v_traces, i_traces, t_trace):
    start_step = np.nonzero(i_traces[0])[0][0]
    end_step = np.nonzero(i_traces[0])[0][-1] + 1
    dur_step = t_trace[end_step] - t_trace[start_step]

    amps = np.array([i_inj[start_step] for i_inj in i_traces])

    firing_rates = np.zeros(len(amps))
    for i, amp in enumerate(amps):
        AP_onsets = get_AP_onset_idxs(v_traces[i], threshold=0)
        n_APs = len(AP_onsets)
        firing_rates[i] = n_APs / dur_step * 1000  # convert to Hz
    return amps, firing_rates


def compute_fIcurve_last_ISI(v_traces, i_traces, t_trace):
    start_step = np.nonzero(i_traces[0])[0][0]
    amps = np.array([i_inj[start_step] for i_inj in i_traces])
    dt = t_trace[1] - t_trace[0]

    firing_rates = np.zeros(len(amps))
    for i, amp in enumerate(amps):
        AP_onsets = get_AP_onset_idxs(v_traces[i], threshold=0)
        if len(AP_onsets) >= 2:
            firing_rates[i] = AP_onsets[-1] * dt - AP_onsets[-2] * dt
        else:
            firing_rates[i] = np.nan
    return amps, firing_rates
