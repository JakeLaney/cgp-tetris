import numpy as np
import scipy.stats

from cgp.functions.support import is_scalar
from cgp.functions.support import is_np
from cgp.functions.support import min_dim

FUNCTIONS = []
FUNC_DESCRIPTIONS = []


def split_before(x, y, p):
    if is_scalar(x):
        return x
    else:
        splitIndex = int((x.shape[0] - 1) * (p + 1) / 2.0)
        return np.copy(x[:splitIndex + 1])
FUNCTIONS.append(split_before)
FUNC_DESCRIPTIONS.append('SPLIT_BEFORE')

def split_after(x, y, p):
    if is_scalar(x):
        return x
    else:
        splitIndex = int((x.shape[0] - 1) * (p + 1) / 2.0)
        return np.copy(x[splitIndex:])
FUNCTIONS.append(split_after)
FUNC_DESCRIPTIONS.append('SPLIT_AFTER')


def range_in(x, y, p):
    if is_scalar(x):
        return x
    else:
        splitY = int((x.shape[0] - 1) * (y + 1) / 2.0)
        splitP = int((x.shape[0] - 1) * (p + 1) / 2.0)
        return np.copy(x[splitY:splitP + 1])
FUNCTIONS.append(range_in)
FUNC_DESCRIPTIONS.append('RANGE_IN')


def index_y(x, y, p):
    if is_scalar(x):
        return x
    else:
        splitY = int((x.shape[0] - 1) * (y + 1) / 2.0)
        return np.copy(x[splitY])
FUNCTIONS.append(index_y)
FUNC_DESCRIPTIONS.append('INDEX_Y')


def index_p(x, y, p):
    if is_scalar(x):
        return x
    else:
        splitP = int((x.shape[0] - 1) * (p + 1) / 2.0)
        return np.copy(x[splitP])
FUNCTIONS.append(index_p)
FUNC_DESCRIPTIONS.append('INDEX_P')


def vectorize(x, y, p):
    if is_scalar(x):
        return x
    else:
        return np.copy(x.flatten())
FUNCTIONS.append(vectorize)
FUNC_DESCRIPTIONS.append('VECTORIZE')


def f_first(x, y, p):
    if is_scalar(x):
        return x
    else:
        return x.flatten()[0]
FUNCTIONS.append(f_first)
FUNC_DESCRIPTIONS.append('FIRST')


def f_last(x, y, p):
    if is_scalar(x):
        return x
    else:
        z = x.flatten()
        return z[len(z) - 1]
FUNCTIONS.append(f_last)
FUNC_DESCRIPTIONS.append('LAST')


def differences(x, y, p):
    if is_scalar(x):
        return x
    else:
        z = x.flatten()
        return np.copy(np.diff(x.flatten()))
FUNCTIONS.append(differences)
FUNC_DESCRIPTIONS.append('DIFFERENCES')


def avg_differences(x, y, p):
    return np.mean(differences(x, y, p))
FUNCTIONS.append(avg_differences)
FUNC_DESCRIPTIONS.append('AVG_DIFFERENCES')


def rotate(x, y, p):
    if is_scalar(x):
        return x
    else:
        return np.roll(x, int(np.ceil(p)))
FUNCTIONS.append(rotate)
FUNC_DESCRIPTIONS.append('ROTATE')


def reverse(x, y, p):
    if not is_np(x):
        return x
    else:
        return np.copy(x[::-1])
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

def set_x(x, y, p):
    return np.mean(x) * np.ones(np.array(y).size)
FUNCTIONS.append(set_x)
FUNC_DESCRIPTIONS.append('SET_X')


def set_y(x, y, p):
    return np.mean(y) * np.ones(np.array(x).size)
FUNCTIONS.append(set_y)
FUNC_DESCRIPTIONS.append('SET_Y')


def sum(x, y, p):
    return np.sum(x)
FUNCTIONS.append(sum)
FUNC_DESCRIPTIONS.append('SUM')

def transpose(x, y, p):
    if is_scalar(x):
        return x
    else:
        return np.transpose(x)
FUNCTIONS.append(transpose)
FUNC_DESCRIPTIONS.append('TRANSPOSE')

# if x is scalar, make it a 1-element array
def vec_from_double(x, y, p):
    if is_scalar(x):
        return np.array([x])
    else:
        return x
FUNCTIONS.append(vec_from_double)
FUNC_DESCRIPTIONS.append('VEC_FROM_DOUB')

# MISC Functions
# TODO probably need to move to another file

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
    if is_scalar(x):
        return np.array([p])
    else:
        return np.full(x.shape, p)
FUNCTIONS.append(constvectord)
FUNC_DESCRIPTIONS.append('CONSTVECTORD')


def zeros(x, y, p):
    if is_scalar(x):
        return np.array([0])
    else:
        return np.zeros(x.shape)
FUNCTIONS.append(zeros)
FUNC_DESCRIPTIONS.append('ZEROS')

def ones(x, y, p):
    if is_scalar(x):
        return np.array([1])
    else:
        return np.ones(x.shape)
FUNCTIONS.append(ones)
FUNC_DESCRIPTIONS.append('ONES')

FUNCTIONS_LEN = len(FUNCTIONS)
