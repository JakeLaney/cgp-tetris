import numpy as np
import scipy.stats

from cgp.functions.support import is_np
from cgp.functions.support import min_dim

FUNCTIONS =[]
FUNCTION_NAMES = []


def stddev(x, y, p):
    if is_np(x):
        return np.std(x)
    else:
        return x
FUNCTIONS.append(stddev)
FUNCTION_NAMES.append('STDDEV')


def skew(x, y, p):
    if is_np(x):
        return scipy.stats.skew(x)
    else:
        return x
FUNCTIONS.append(skew)
FUNCTION_NAMES.append('SKEW')


def kurtosis(x, y, p):
    if is_np(x):
        return scipy.stats.kurtosis(x)
    else:
        return x
FUNCTIONS.append(kurtosis)
FUNCTION_NAMES.append('KURTOSIS')


def mean(x, y, p):
    return np.mean(x)
FUNCTIONS.append(mean)
FUNCTION_NAMES.append('MEAN')


def f_range(x, y, p):
    if is_np(x):
        return np.max(x) - np.min(x) - 1
    else:
        return x
FUNCTIONS.append(f_range)
FUNCTION_NAMES.append('RANGE')


def f_round(x, y, p):
    return np.round(x)
FUNCTIONS.append(f_round)
FUNCTION_NAMES.append('ROUND')


def f_floor(x, y, p):
    return np.floor(x)
FUNCTIONS.append(f_floor)
FUNCTION_NAMES.append('FLOOR')

def f_ceil(x, y, p):
    return np.ceil(x)
FUNCTIONS.append(f_ceil)
FUNCTION_NAMES.append('CEIL')


def max1(x, y, p):
    return np.max(x)
FUNCTIONS.append(max1)
FUNCTION_NAMES.append('MAX1')

def min1(x, y, p):
    return np.min(x)
FUNCTIONS.append(min1)
FUNCTION_NAMES.append('MIN1')
