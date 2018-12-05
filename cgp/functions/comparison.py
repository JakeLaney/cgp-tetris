import numpy as np
import scipy.stats

from cgp.functions.support import is_numpy_array
from cgp.functions.support import minimum_shape

FUNCTIONS = []
FUNC_DESCRIPTIONS = []


def lt(x, y, p):
    if is_numpy_array(x) and is_numpy_array(y):
        new_dim = minimum_shape(x, y)
        x = np.resize(x, new_dim)
        y = np.resize(y, new_dim)
    return x < y


FUNCTIONS.append(lt)
FUNC_DESCRIPTIONS.append('LT')


def gt(x, y, p):
    if is_numpy_array(x) and is_numpy_array(y):
        new_dum = minimum_shape(x, y)
        x = np.resize(x, new_dum)
        y = np.resize(y, new_dum)
    return x > y


FUNCTIONS.append(gt)
FUNC_DESCRIPTIONS.append('GT')


def max2(x, y, p):
    if is_numpy_array(x) and is_numpy_array(y):
        new_dim = minimum_shape(x, y)
        x = np.resize(x, new_dim)
        y = np.resize(y, new_dim)
    return np.maximum(x, y)


FUNCTIONS.append(max2)
FUNC_DESCRIPTIONS.append('MAX2')


def min2(x, y, p):
    if is_numpy_array(x) and is_numpy_array(y):
        new_dim = minimum_shape(x, y)
        x = np.resize(x, new_dim)
        y = np.resize(y, new_dim)
    return np.minimum(x, y)


# TODO: 'FUNCTIONS.append(min2)' was missing. is this on purpose?
# FUNCTIONS.append(min2)
# FUNC_DESCRIPTIONS.append('MIN2')
