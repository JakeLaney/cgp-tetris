
from config import Config
from cgp.functionset import FunctionSet
from cgp.genome import Genome

import numpy.random 

def main():
    config = Config()
    functionSet = FunctionSet()
    genome = Genome(config, functionSet)
    r = numpy.random.rand(5, 5)
    g = numpy.random.rand(5, 5)
    b = numpy.random.rand(5, 5)
    print genome.evaluate(r, g, b)

if __name__ == '__main__':
    main()
