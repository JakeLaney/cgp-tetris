import numpy as np
import scipy.stats

from cgp.functions.support import is_scalar
from cgp.functions.support import is_np
from cgp.functions.support import min_dim

FUNCTIONS = []
FUNCTION_NAMES = []


def split_before(x, y, p):
    if is_scalar(x):
        return x
    else:
        splitIndex = int((x.shape[0] - 1) * (p + 1) / 2.0)
        return np.copy(x[:splitIndex + 1])
FUNCTIONS.append(split_before)
FUNCTION_NAMES.append('SPLIT_BEFORE')

def split_after(x, y, p):
    if is_scalar(x):
        return x
    else:
        splitIndex = int((x.shape[0] - 1) * (p + 1) / 2.0)
        return np.copy(x[splitIndex:])
FUNCTIONS.append(split_after)
FUNCTION_NAMES.append('SPLIT_AFTER')


def range_in(x, y, p):
    if is_scalar(x) or not is_scalar(y):
        return x
    else:
        splitY = int((x.shape[0] - 1) * (y + 1) / 2.0)
        splitP = int((x.shape[0] - 1) * (p + 1) / 2.0)
        start = min(splitY, splitP)
        end = max(splitY, splitP)
        return np.copy(x[start:end + 1])
FUNCTIONS.append(range_in)
FUNCTION_NAMES.append('RANGE_IN')


def index_y(x, y, p):
    if is_scalar(x) or not is_scalar(y):
        return x
    else:
        splitY = int((x.shape[0] - 1) * (y + 1) / 2.0)
        return np.copy(x[splitY])
FUNCTIONS.append(index_y)
FUNCTION_NAMES.append('INDEX_Y')


def index_p(x, y, p):
    if is_scalar(x):
        return x
    else:
        splitP = int((x.shape[0] - 1) * (p + 1) / 2.0)
        return np.copy(x[splitP])
FUNCTIONS.append(index_p)
FUNCTION_NAMES.append('INDEX_P')


def vectorize(x, y, p):
    if is_scalar(x):
        return x
    else:
        return np.copy(x.flatten())
FUNCTIONS.append(vectorize)
FUNCTION_NAMES.append('VECTORIZE')


def f_first(x, y, p):
    if is_scalar(x):
        return x
    else:
        return x.flatten()[0]
FUNCTIONS.append(f_first)
FUNCTION_NAMES.append('FIRST')


def f_last(x, y, p):
    if is_scalar(x):
        return x
    else:
        z = x.flatten()
        return z[len(z) - 1]
FUNCTIONS.append(f_last)
FUNCTION_NAMES.append('LAST')


def differences(x, y, p):
    if is_scalar(x):
        return x
    else:
        z = x.flatten()
        if z.size > 2:
            return np.copy(np.diff(x.flatten()))
        else:
            return z
FUNCTIONS.append(differences)
FUNCTION_NAMES.append('DIFFERENCES')


def avg_differences(x, y, p):
    return np.mean(differences(x, y, p))
FUNCTIONS.append(avg_differences)
FUNCTION_NAMES.append('AVG_DIFFERENCES')


def rotate(x, y, p):
    if is_scalar(x):
        return x
    else:
        return np.roll(x, int(np.ceil(p)))
FUNCTIONS.append(rotate)
FUNCTION_NAMES.append('ROTATE')


def reverse(x, y, p):
    if not is_np(x):
        return x
    else:
        return np.copy(x[::-1])
FUNCTIONS.append(reverse)
FUNCTION_NAMES.append('REVERSE')


def push_back(x, y, p):
    return np.append(np.array(x).flatten(), np.array(y).flatten())
#FUNCTIONS.append(push_back)
#FUNCTION_NAMES.append('PUSH_BACK')


def push_back2(x, y, p):
    return np.append(np.array(y).flatten(), np.array(x).flatten())
FUNCTIONS.append(push_back2)
FUNCTION_NAMES.append('PUSH_BACK2')

def set_x(x, y, p):
    return np.mean(x) * np.ones(np.array(y).size)
FUNCTIONS.append(set_x)
FUNCTION_NAMES.append('SET_X')


def set_y(x, y, p):
    return np.mean(y) * np.ones(np.array(x).size)
FUNCTIONS.append(set_y)
FUNCTION_NAMES.append('SET_Y')


def sum(x, y, p):
    return np.sum(x)
FUNCTIONS.append(sum)
FUNCTION_NAMES.append('SUM')

def transpose(x, y, p):
    if is_scalar(x):
        return x
    else:
        return np.transpose(x)
FUNCTIONS.append(transpose)
FUNCTION_NAMES.append('TRANSPOSE')

# if x is scalar, make it a 1-element array
def vec_from_double(x, y, p):
    if is_scalar(x):
        return np.array([x])
    else:
        return x
FUNCTIONS.append(vec_from_double)
FUNCTION_NAMES.append('VEC_FROM_DOUB')

# MISC Functions
# TODO probably need to move to another file

def ywire(x, y, p):
    return y
FUNCTIONS.append(ywire)
FUNCTION_NAMES.append('YWIRE')

def nop(x, y, p):
    return x
FUNCTIONS.append(nop)
FUNCTION_NAMES.append('NOP')


def const(x, y, p):
    return p
FUNCTIONS.append(const)
FUNCTION_NAMES.append('CONST')


def constvectord(x, y, p):
    if is_scalar(x):
        return np.array([p])
    else:
        return np.full(x.shape, p)
FUNCTIONS.append(constvectord)
FUNCTION_NAMES.append('CONSTVECTORD')


def zeros(x, y, p):
    if is_scalar(x):
        return np.array([0])
    else:
        return np.zeros(x.shape)
FUNCTIONS.append(zeros)
FUNCTION_NAMES.append('ZEROS')

def ones(x, y, p):
    if is_scalar(x):
        return np.array([1])
    else:
        return np.ones(x.shape)
FUNCTIONS.append(ones)
FUNCTION_NAMES.append('ONES')

FUNCTIONS_LEN = len(FUNCTIONS)
