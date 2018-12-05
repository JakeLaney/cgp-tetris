import numpy as np
import scipy.stats

from cgp.functions.support import is_np
from cgp.functions.support import min_dim

FUNCTIONS = []
FUNC_DESCRIPTIONS = []


def lt(x, y, p):
    if is_np(x) and is_np(y):
        new_dim = min_dim(x, y)
        x = np.resize(x, new_dim)
        y = np.resize(y, new_dim)
    return x < y


FUNCTIONS.append(lt)
FUNC_DESCRIPTIONS.append('LT')


def gt(x, y, p):
    if is_np(x) and is_np(y):
        new_dum = min_dim(x, y)
        x = np.resize(x, new_dum)
        y = np.resize(y, new_dum)
    return x > y


FUNCTIONS.append(gt)
FUNC_DESCRIPTIONS.append('GT')


def max2(x, y, p):
    if is_np(x) and is_np(y):
        new_dim = min_dim(x, y)
        x = np.resize(x, new_dim)
        y = np.resize(y, new_dim)
    return np.maximum(x, y)


FUNCTIONS.append(max2)
FUNC_DESCRIPTIONS.append('MAX2')


def min2(x, y, p):
    if is_np(x) and is_np(y):
        new_dim = min_dim(x, y)
        x = np.resize(x, new_dim)
        y = np.resize(y, new_dim)
    return np.minimum(x, y)


FUNCTIONS.append(min2)
FUNC_DESCRIPTIONS.append('MIN2')
