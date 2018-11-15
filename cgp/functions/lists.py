import numpy as np
import scipy.stats

from cgp.functions.support import is_numpy_array
from cgp.functions.support import minimum_shape

FUNCTIONS = []

def split_before(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        splitIndex = int(np.floor((p + 1) / 2))
        return np.copy(x[:splitIndex])


FUNCTIONS.append(split_before)


def split_after(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        value = (p + 1) / 2.0
        index = int(np.floor(value * x.shape[0]))
        return np.copy(x[index:])


FUNCTIONS.append(split_after)


def range_in(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        valueY = (np.mean(y) + 1) / 2.0
        valueP = (p + 1) / 2.0
        start = int(np.floor(valueY) % x.shape[0])
        end = int(np.floor(valueP * x.shape[0]))
        return np.copy(x[start:end])


FUNCTIONS.append(range_in)

# TODO index_y


def index_y(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        value = (np.mean(y) + 1) / 2.0
        index = int(np.floor(value) % x.shape[0])
        return x[index]


FUNCTIONS.append(index_y)


def index_p(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        value = (p + 1) / 2.0
        index = int(np.floor(x.shape[0] * p))
        return x[index]


FUNCTIONS.append(index_p)


def vectorize(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return x.flatten()


FUNCTIONS.append(vectorize)


def first(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return x.flatten()[0]


FUNCTIONS.append(first)


def f_last(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        z = x.flatten()
        return int(z[x.size - 1])


FUNCTIONS.append(f_last)

# TODO differences
# TODO avg_differences


def rotate(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return np.roll(x, int(np.ceil(p)))


FUNCTIONS.append(rotate)


def reverse(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return x


FUNCTIONS.append(reverse)


def push_back(x, y, p):
    return np.append(np.array(x).flatten(), np.array(y).flatten())


FUNCTIONS.append(push_back)


def push_back2(x, y, p):
    return np.append(np.array(y).flatten(), np.array(x).flatten())


FUNCTIONS.append(push_back2)

# TODO set(x, y, p)


def sum(x, y, p):
    return np.sum(x)


FUNCTIONS.append(sum)

# TODO tranpose
# TODO vecfromdouble


def ywire(x, y, p):
    return y


FUNCTIONS.append(ywire)


def nop(x, y, p):
    return x


FUNCTIONS.append(nop)


def const(x, y, p):
    return p


FUNCTIONS.append(const)


def constvectord(x, y, p):
    return np.full(np.array(x).shape, p)


FUNCTIONS.append(constvectord)


def zeros(x, y, p):
    return np.zeros(np.array(x).shape)


FUNCTIONS.append(zeros)


def ones(x, y, p):
    return np.ones(np.array(x).shape)


FUNCTIONS.append(ones)


FUNCTIONS_LEN = len(FUNCTIONS)
