
from math import floor
from random import random
from random import randint
import numpy as np
from cgp.functions.support import is_scalar
from cgp.functions.support import is_np

class Gene:
    def __init__(self, config, index):
        self.functionSet = config.functionSet
        self.functionSetLen = len(config.functionSet)
        self.scalar = config.inputScalarR
        self.totalGenes = config.get_genome_size()
        self.n = index
        self.fraction = self.n / float(self.totalGenes)
        self.output = 0
        self.active = False
        self.mutate()

        # used in functionset.py to keep track of which genes have been used
        self.active_in_functional_graph = False

    def copy_into(self, other):
        other.n = self.n
        other.fraction = self.fraction
        other.output = self.output
        other.active = self.active
        other.x = self.x
        other.y = self.y
        other.f = self.f
        other.p = self.p

    def rand(self):
        return uniform(0.0, 1.0)

    def mutate(self):
        self.output = 0
        self.x = random()
        self.y = random()
        self.f = random()
        self.p = random()

    def randIndex(self, array):
        return randint(0, len(array) - 1)

    # TODO: resolve issues with get_x, get_y returning zero
    def getx(self):
        return int(round(self.x * (self.totalGenes - 1)))

    def gety(self):
        return int(round(self.y * (self.totalGenes - 1)))

    def get_x(self):
        fraction = self.n / float(self.totalGenes)
        x = int(floor(self.x * ((1 - fraction) * self.scalar + fraction) * self.totalGenes))
        return x

    def get_y(self):
        fraction = self.n / float(self.totalGenes)
        return int(floor(self.y * ((1 - fraction) * self.scalar + fraction) * self.totalGenes))

    def get_p(self):
        return (2.0 * self.p) - 1

    def get_f(self):
        index = int(round(self.f * (self.functionSetLen - 1)))
        return index

    def get_function(self):
        return self.functionSet[self.get_f()]

    def init_as_input_gene(self, inputValue):
        self.output = inputValue

    def prepare_for_evaluation(self):
        self.active = False

    def evaluate(self, x, y, p):
        function = self.get_function()
        self.output = self.constrain_result(p * function(x, y, p))

    def constrain_result(self, result):
        if is_scalar(result):
            if result > 1.0:
                return 1.0
            elif result < -1.0:
                return -1.0
            else:
                return result
        else:
            result[result > 1.0] = 1.0
            result[result < -1.0] = -1.0
            return result
