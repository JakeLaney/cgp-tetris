import numpy as np
import scipy.stats

from cgp.functions.support import is_np
from cgp.functions.support import min_dim

FUNCTIONS = []
FUNCTION_NAMES = []


def lt(x, y, p):
    if is_np(x) and is_np(y):
        new_dim = min_dim(x, y)
        return np.resize(x, new_dim) < np.resize(y, new_dim)
    return x < y
FUNCTIONS.append(lt)
FUNCTION_NAMES.append('LT')


def gt(x, y, p):
    if is_np(x) and is_np(y):
        new_dim = min_dim(x, y)
        return np.resize(x, new_dim) > np.resize(y, new_dim)
    return x > y
FUNCTIONS.append(gt)
FUNCTION_NAMES.append('GT')


def max2(x, y, p):
    return np.max([np.max(x), np.max(y)])
FUNCTIONS.append(max2)
FUNCTION_NAMES.append('MAX2')


def min2(x, y, p):
        return np.min([np.min(x), np.min(y)])
FUNCTIONS.append(min2)
FUNCTION_NAMES.append('MIN2')
