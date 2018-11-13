
from config import Config
from cgp.functionset import FunctionSet
from cgp.genome import Genome

import numpy as np
import numpy.random

from PIL import Image

SIZE = 250

def main():
    config = Config()
    functionSet = FunctionSet()
    r = numpy.random.rand(SIZE, SIZE)
    g = numpy.random.rand(SIZE, SIZE)
    b = numpy.random.rand(SIZE, SIZE)
    print 'start...'
    genome = Genome(config, functionSet)
    print np.max(genome.evaluate(r, g, b))

if __name__ == '__main__':
    main()
