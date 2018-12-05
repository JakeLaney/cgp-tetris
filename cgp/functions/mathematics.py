import numpy as np
import scipy.stats

from cgp.functions.support import is_scalar
from cgp.functions.support import is_np
from cgp.functions.support import min_dim

FUNCTIONS = []
FUNC_DESCRIPTIONS = []


def add(x, y, p):
    if is_np(x) and is_np(y):
        new_dim = min_dim(x, y)
        return (np.resize(x, new_dim) + np.resize(y, new_dim)) / 2.0
    return (x + y) / 2.0
FUNCTIONS.append(add)
FUNC_DESCRIPTIONS.append('ADD')


def aminus(x, y, p):
    if is_np(x) and is_np(y):
        new_dim = min_dim(x, y)
        return np.abs(np.resize(x, new_dm) - np.resize(y, new_dm)) / 2.0
    return np.abs(x - y) / 2.0
FUNCTIONS.append(aminus)
FUNC_DESCRIPTIONS.append('AMINUS')


def mult(x, y, p):
    if is_np(x) and is_np(y):
        new_dim = min_dim(x, y)
        return np.resize(x, new_dim) * np.resize(y, new_dim)
    return x * y
FUNCTIONS.append(mult)
FUNC_DESCRIPTIONS.append('MULT')


def cmult(x, y, p):
    return x * p
FUNCTIONS.append(cmult)
FUNC_DESCRIPTIONS.append('CMULT')


# TODO check this again
def inv(x, y, p):
    if is_np(x):
        out = np.zeros(x.shape)
        np.divide(1.0, x, out=out, where=(x != 0.0))
        return out
    else:
        return x if x == 0 else 1.0 / x
FUNCTIONS.append(inv)
FUNC_DESCRIPTIONS.append('INV')


def abs(x, y, p):
    return np.abs(x)
FUNCTIONS.append(abs)
FUNC_DESCRIPTIONS.append('ABS')


def sqrt(x, y, p):
    return np.sqrt(np.abs(x))
FUNCTIONS.append(sqrt)
FUNC_DESCRIPTIONS.append('SQRT')


def cpow(x, y, p):
    r = np.abs(x) ** (p + 1)
    return r
FUNCTIONS.append(cpow)
FUNC_DESCRIPTIONS.append('CPOW')


def ypow(x, y, p):
    return np.abs(x) ** np.abs(y)
FUNCTIONS.append(ypow)
FUNC_DESCRIPTIONS.append('YPOW')


def expx(x, y, p):
    return (np.exp(x) - 1) / (np.exp(1) - 1)
FUNCTIONS.append(expx)
FUNC_DESCRIPTIONS.append('EXPX')


def sinx(x, y, p):
    return np.sin(x)
FUNCTIONS.append(sinx)
FUNC_DESCRIPTIONS.append('SINX')


def sqrtxy(x, y, p):
    return np.sqrt(np.square(x) + np.square(y)) / np.sqrt(2.0)
FUNCTIONS.append(sqrtxy)
FUNC_DESCRIPTIONS.append('SQRTXY')


def acos(x, y, p):
    return np.arccos(x) / np.pi
FUNCTIONS.append(acos)
FUNC_DESCRIPTIONS.append('ACOS')


def asin(x, y, p):
    return 2.0 * np.arcsin(x) / np.pi
FUNCTIONS.append(asin)
FUNC_DESCRIPTIONS.append('ASIN')


def atan(x, y, p):
    return 4.0 * np.arctan(x) / np.pi
FUNCTIONS.append(atan)
FUNC_DESCRIPTIONS.append('ATAN')
