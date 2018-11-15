import numpy as np
import scipy.stats

from cgp.functions.support import is_numpy_array
from cgp.functions.support import minimum_shape

FUNCTIONS = []

def add(x, y, p):
    if (is_numpy_array(x) and is_numpy_array(y)):
        newDim = minimum_shape(x, y)
        x = np.resize(x, newDim)
        y = np.resize(y, newDim)
    return x + y / 2.0
FUNCTIONS.append(add)


def aminus(x, y, p):
    if (is_numpy_array(x) and is_numpy_array(y)):
        newDim = minimum_shape(x, y)
        x = np.resize(x, newDim)
        y = np.resize(y, newDim)
    return np.abs(x - y) / 2.0
FUNCTIONS.append(aminus)


def mult(x, y, p):
    if (is_numpy_array(x) and is_numpy_array(y)):
        newDim = minimum_shape(x, y)
        x = np.resize(x, newDim)
        y = np.resize(y, newDim)
    return x * y


FUNCTIONS.append(mult)


def cmult(x, y, p):
    return x * p


FUNCTIONS.append(cmult)


def inv(x, y, p):
    if is_numpy_array(x):
        value = 1.0 / x
        value[x == np.inf] = 0
        value[x == -np.inf] = 0
        return value
    else:
        if x == 0:
            return 0
        else:
            return 1.0 / x

FUNCTIONS.append(inv)


def abs(x, y, p):
    return np.abs(x)


FUNCTIONS.append(abs)


def sqrt(x, y, p):
    return np.sqrt(np.abs(x))


FUNCTIONS.append(sqrt)


def cpow(x, y, p):
    return np.abs(x) ** (p + 1)


FUNCTIONS.append(cpow)


def ypow(x, y, p):
    return np.abs(x) ** np.abs(y)


FUNCTIONS.append(ypow)


def expx(x, y, p):
    return (np.exp(x) - 1) / (np.exp(1) - 1)


FUNCTIONS.append(expx)


def sinx(x, y, p):
    return np.sin(x)


FUNCTIONS.append(sinx)


def sqrtxy(x, y, p):
    return np.sqrt(np.square(x) + np.square(y)) / np.sqrt(2.0)


FUNCTIONS.append(sqrtxy)


def acos(x, y, p):
    return np.arccos(x) / np.pi


FUNCTIONS.append(acos)


def asin(x, y, p):
    return 2.0 * np.arcsin(x) / np.pi


FUNCTIONS.append(asin)


def atan(x, y, p):
    return 4.0 * np.arctan(x) / np.pi


FUNCTIONS.append(atan)
