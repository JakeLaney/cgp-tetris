import numpy as np
import scipy.stats

from cgp.functions.support import is_numpy_array
from cgp.functions.support import minimum_shape

FUNCTIONS = []
FUNC_DESCRIPTIONS = []

def split_before(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        splitIndex = int(np.floor((p + 1) / 2))
        return np.copy(x[:splitIndex])


FUNCTIONS.append(split_before)
FUNC_DESCRIPTIONS.append('SPLIT_BEFORE')


def split_after(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        value = (p + 1) / 2.0
        index = int(np.floor(value * x.shape[0]))
        return np.copy(x[index:])


FUNCTIONS.append(split_after)
FUNC_DESCRIPTIONS.append('SPLIT_AFTER')


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
FUNC_DESCRIPTIONS.append('RANGE_IN')


# TODO index_y
def index_y(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        value = (np.mean(y) + 1) / 2.0
        index = int(np.floor(value) % x.shape[0])
        return x[index]


FUNCTIONS.append(index_y)
FUNC_DESCRIPTIONS.append('INDEX_Y')


def index_p(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        value = (p + 1) / 2.0
        index = int(np.floor(x.shape[0] * p))
        return x[index]


FUNCTIONS.append(index_p)
FUNC_DESCRIPTIONS.append('INDEX_P')


def vectorize(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return x.flatten()


FUNCTIONS.append(vectorize)
FUNC_DESCRIPTIONS.append('VECTORIZE')


def first(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return x.flatten()[0]


FUNCTIONS.append(first)
FUNC_DESCRIPTIONS.append('FIRST')


def f_last(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        z = x.flatten()
        return int(z[x.size - 1])


FUNCTIONS.append(f_last)
FUNC_DESCRIPTIONS.append('LAST')

# TODO differences
# TODO avg_differences


def rotate(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return np.roll(x, int(np.ceil(p)))


FUNCTIONS.append(rotate)
FUNC_DESCRIPTIONS.append('ROTATE')


def reverse(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return x


FUNCTIONS.append(reverse)
FUNC_DESCRIPTIONS.append('REVERSE')


def push_back(x, y, p):
    return np.append(np.array(x).flatten(), np.array(y).flatten())


FUNCTIONS.append(push_back)
FUNC_DESCRIPTIONS.append('PUSH_BACK')


def push_back2(x, y, p):
    return np.append(np.array(y).flatten(), np.array(x).flatten())


FUNCTIONS.append(push_back2)
FUNC_DESCRIPTIONS.append('PUSH_BACK2')

# TODO set(x, y, p)


def sum(x, y, p):
    return np.sum(x)


FUNCTIONS.append(sum)
FUNC_DESCRIPTIONS.append('SUM')

# TODO transpose
# TODO vecfromdouble


def ywire(x, y, p):
    return y


FUNCTIONS.append(ywire)
FUNC_DESCRIPTIONS.append('YWIRE')


def nop(x, y, p):
    return x


FUNCTIONS.append(nop)
FUNC_DESCRIPTIONS.append('NOP')


def const(x, y, p):
    return p


FUNCTIONS.append(const)
FUNC_DESCRIPTIONS.append('CONST')


def constvectord(x, y, p):
    return np.full(np.array(x).shape, p)


FUNCTIONS.append(constvectord)
FUNC_DESCRIPTIONS.append('CONSTVECTORD')


def zeros(x, y, p):
    return np.zeros(np.array(x).shape)


FUNCTIONS.append(zeros)
FUNC_DESCRIPTIONS.append('ZEROS')


def ones(x, y, p):
    return np.ones(np.array(x).shape)


FUNCTIONS.append(ones)
FUNC_DESCRIPTIONS.append('ONES')


FUNCTIONS_LEN = len(FUNCTIONS)
