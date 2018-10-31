from __future__ import division
import numpy as np

__author__ = 'caro'


def to_idx(time_point, dt, decimal_place=None):
    idx = time_point/dt
    if decimal_place:
        assert np.round(idx * dt, decimal_place) == np.round(time_point, decimal_place), \
            'Time points given are not uniquely identifiable given dt. ' \
            'time_point: %f, dt: %f' % (time_point, dt)
        return int(np.round(idx, 0))  # rounding necessary because conversion to int sometimes weird
    else:
        assert idx * dt == time_point, 'Time points given are not uniquely identifiable given dt. ' \
                                       'time_point: %f, dt: %f' % (time_point, dt)
        return int(np.round(idx, 0))  # rounding necessary because conversion to int sometimes weird


def exp_fit(t, a, v):
    diff_exp = np.max(np.exp(-t / a)) - np.min(np.exp(-t / a))
    diff_points = v[0] - v[-1]
    return (np.exp(-t / a) - (np.exp(-t / a))[0]) / diff_exp * diff_points + v[0]