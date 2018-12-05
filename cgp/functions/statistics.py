import numpy as np
import scipy.stats

from cgp.functions.support import is_np
from cgp.functions.support import min_dim

FUNCTIONS =[]
FUNC_DESCRIPTIONS = []


def stddev(x, y, p):
    if is_np(x):
        return np.std(x)
    else:
        return x


FUNCTIONS.append(stddev)
FUNC_DESCRIPTIONS.append('STDDEV')


def skew(x, y, p):
    if is_np(x):
        return scipy.stats.skew(x)
    else:
        return x


FUNCTIONS.append(skew)
FUNC_DESCRIPTIONS.append('SKEW')


def kurtosis(x, y, p):
    if is_np(x):
        return scipy.stats.kurtosis(x)
    else:
        return x


FUNCTIONS.append(kurtosis)
FUNC_DESCRIPTIONS.append('KURTOSIS')


def mean(x, y, p):
    return np.mean(x)


FUNCTIONS.append(mean)
FUNC_DESCRIPTIONS.append('MEAN')


def f_range(x, y, p):
    if is_np(x):
        return np.max(x) - np.min(x) - 1
    else:
        return x


FUNCTIONS.append(f_range)
FUNC_DESCRIPTIONS.append('RANGE')


def f_round(x, y, p):
    return np.round(x)


FUNCTIONS.append(f_round)
FUNC_DESCRIPTIONS.append('ROUND')


def f_floor(x, y, p):
    return np.floor(x)


FUNCTIONS.append(f_floor)
FUNC_DESCRIPTIONS.append('FLOOR')


def max1(x, y, p):
    return np.max(x)


FUNCTIONS.append(max1)
FUNC_DESCRIPTIONS.append('MAX1')


def min1(x, y, p):
    return np.min(x)


FUNCTIONS.append(min1)
FUNC_DESCRIPTIONS.append('MIN1')
