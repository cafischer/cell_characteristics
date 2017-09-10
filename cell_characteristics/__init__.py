from __future__ import division
import numpy as np

__author__ = 'caro'


def to_idx(time_point, dt):
    idx = time_point/dt
    assert idx * dt == time_point, 'Time points given are not uniquely identifiable given dt.'
    return int(idx)


def exp_fit(t, a, v):
    diff_exp = np.max(np.exp(-t / a)) - np.min(np.exp(-t / a))
    diff_points = v[0] - v[-1]
    return (np.exp(-t / a) - (np.exp(-t / a))[0]) / diff_exp * diff_points + v[0]