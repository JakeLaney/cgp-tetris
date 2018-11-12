import numpy as np
import scipy.stats

from support import is_numpy_array
from support import minimum_shape

FUNCTIONS =[]

def stddev(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return np.std(x)


FUNCTIONS.append(stddev)


def skew(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return scipy.stats.skew(x)


FUNCTIONS.append(skew)


def kurtosis(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return scipy.stats.kurtosis(x)


FUNCTIONS.append(kurtosis)


def mean(x, y, p):
    return np.mean(x)


FUNCTIONS.append(mean)


def f_range(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return np.max(x) - np.min(x) - 1


FUNCTIONS.append(f_range)


def f_round(x, y, p):
    return np.round(x)


FUNCTIONS.append(f_round)


def f_floor(x, y, p):
    return np.floor(x)


FUNCTIONS.append(f_floor)


def max1(x, y, p):
    return np.max(x)


FUNCTIONS.append(max1)


def min1(x, y, p):
    return np.min(x)


FUNCTIONS.append(min1)
