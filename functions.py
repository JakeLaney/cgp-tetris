import numpy as np 
import scipy.stats

from support import is_numpy_array
from support import minimum_shape

FUNCTIONS = []

#### 1. MATHEMATICAL FUNCTIONS ####

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
    value = 1.0 / x
    value[x == np.inf] = 0
    value[x == -np.inf] = 0
    return value
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
    return np.arcos(x) / np.pi
FUNCTIONS.append(acos)

def asin(x, y, p):
    return 2.0 * np.arcsin(x) / np.pi
FUNCTIONS.append(asin)

def atan(x, y, p):
    return 4.0 * np.arctan(x) / np.pi
FUNCTIONS.append(atan)

#### 2. Statistical FUNCTIONS ####

def stddev(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return np.std(x)
FUNCTIONS.append(stddev)

def skew(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return scipy.stats.skew(x)
FUNCTIONS.append(skew)

def kurtosis(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return scipy.stats.kurtosis(x) 
FUNCTIONS.append(kurtosis)

def mean(x, y, p):
    return np.mean(x)
FUNCTIONS.append(mean)

def f_range(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return np.max(x) - np.min(x) - 1
FUNCTIONS.append(f_range)

def f_round(x, y, p):
    return np.round(x)
FUNCTIONS.append(f_round)

def f_floor(x, y, p):
    return np.floor(x)
FUNCTIONS.append(f_floor)

def max1(x, y, p):
    return np.max(x)
FUNCTIONS.append(max1)

def min1(x, y, p):
    return np.min(x)
FUNCTIONS.append(min1)

#### 3. Comparison ####

def lt(x, y, p):
    if (is_numpy_array(x) and is_numpy_array(y)):
        newDim = minimum_shape(x, y)
        x = np.resize(x, newDim)
        y = np.resize(y, newDim)
    return x < y
FUNCTIONS.append(lt)

def gt(x, y, p):
    if (is_numpy_array(x) and is_numpy_array(y)):
        newDim = minimum_shape(x, y)
        x = np.resize(x, newDim)
        y = np.resize(y, newDim)
    return x > y 
FUNCTIONS.append(gt)

def max2(x, y, p):
    if (is_numpy_array(x) and is_numpy_array(y)):
        newDim = minimum_shape(x, y)
        x = np.resize(x, newDim)
        y = np.resize(y, newDim)
    return np.maximum(x, y)
FUNCTIONS.append(max2)

def min2(x, y, p):
    if (is_numpy_array(x) and is_numpy_array(y)):
        newDim = minimum_shape(x, y)
        x = np.resize(x, newDim)
        y = np.resize(y, newDim)
    return np.minimum(x, y)

#### 4. Lists ####

def split_before(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        splitIndex = np.floor((p + 1) / 2)
        return np.copy(x[:splitIndex])
FUNCTIONS.append(split_before)

def split_after(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        value = (p + 1) / 2.0
        index = np.floor(value * x.size)
        return np.copy(x[index:])
FUNCTIONS.append(split_after)

def range_in(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        valueY = (np.mean(y) + 1) / 2.0
        valueP = (p + 1) / 2.0
        start =  np.floor(valueY) % x.size
        end = np.floor(valueP * x.size)
        return np.copy(x[start:end])
FUNCTIONS.append(range_in)

# TODO index_y
def index_y(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        value = (np.mean(y) + 1) / 2.0
        index = np.floor(value) % x.size
        return x[index]
FUNCTIONS.append(index_y)

def index_p(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        value = (p + 1) / 2.0
        index = np.floor(x.size * p)
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

def last(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        z = x.flatten()
        return z[x.size - 1]
FUNCTIONS.append(last)

# TODO differences
# TODO avg_differences

def rotate(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return np.roll(x, np.ceil(p))
FUNCTIONS.append(rotate)

def reverse(x, y, p):
    if not is_numpy_array(x):
        return x
    else:
        return np.flip(x)
FUNCTIONS.append(reverse)

def push_back(x, y, p):
    return np.concatenate(numpy.array(x).flatten(), numpy.array(y).flatten())
FUNCTIONS.append(push_back)

def push_back2(x, y, p):
    return np.concatenate(numpy.array(y).flatten(), numpy.array(x).flatten())
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
    return np.full(x.shape, p)
FUNCTIONS.append(constvectord)

def zeros(x, y, p):
    return np.zeros(x.shape)
FUNCTIONS.append(zeros)

def ones(x, y, p):
    return np.ones(x.shape)
FUNCTIONS.append(ones)







