
from numpy import ndarray

def is_scalar(obj):
    return not is_np(obj)

def is_matrix(obj):
    return hasattr(type(obj), '__len__') and hasattr(type(obj), '__getitem__')

def is_np(arr):
    return type(arr) == ndarray and arr.ndim != 0

def min_dim(a, b):
    aDims = len(a.shape)
    bDims = len(b.shape)

    minDims = []

    if aDims < bDims:
        for i in range(aDims):
            minDims.append(min(a.shape[i], b.shape[i]))
    else:
        for i in range(bDims):
            minDims.append(min(a.shape[i], b.shape[i]))

    return minDims
