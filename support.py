
from numpy import ndarray

def is_numpy_array(arr):
    return type(arr) == ndarray

def minimum_shape(numpyA, numpyB):
    aR, aC = numpyA.shape
    bR, bC = numpyB.shape
    minR = minimum(aR, bR)
    minC = minimum(aC, bC)
    return (minR, minC)

def minimum(a, b): 
    if a < b:
        return a
    else:
        return b