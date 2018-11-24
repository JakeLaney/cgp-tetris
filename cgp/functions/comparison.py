import numpy as np
import scipy.stats

from cgp.functions.support import is_numpy_array
from cgp.functions.support import minimum_shape

FUNCTIONS = []
FUNC_DESCRIPTIONS = []


def lt(x, y, p):
    if (is_numpy_array(x) and is_numpy_array(y)):
        newDim = minimum_shape(x, y)
        x = np.resize(x, newDim)
        y = np.resize(y, newDim)
    return x < y


FUNCTIONS.append(lt)
FUNC_DESCRIPTIONS.append('LT')


def gt(x, y, p):
    if (is_numpy_array(x) and is_numpy_array(y)):
        newDim = minimum_shape(x, y)
        x = np.resize(x, newDim)
        y = np.resize(y, newDim)
    return x > y


FUNCTIONS.append(gt)
FUNC_DESCRIPTIONS.append('GT')


def max2(x, y, p):
    if (is_numpy_array(x) and is_numpy_array(y)):
        newDim = minimum_shape(x, y)
        x = np.resize(x, newDim)
        y = np.resize(y, newDim)
    return np.maximum(x, y)


FUNCTIONS.append(max2)
FUNC_DESCRIPTIONS.append('MAX2')


def min2(x, y, p):
    if (is_numpy_array(x) and is_numpy_array(y)):
        newDim = minimum_shape(x, y)
        x = np.resize(x, newDim)
        y = np.resize(y, newDim)
    return np.minimum(x, y)


# TODO: 'FUNCTIONS.append(min2)' was missing. is this on purpose?
# FUNCTIONS.append(min2)
# FUNC_DESCRIPTIONS.append('MIN2')
