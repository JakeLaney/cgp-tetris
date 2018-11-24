
from math import floor
from random import uniform
from random import randint
import numpy as np

class Gene:
    def __init__(self, config, functionSet, index):
        self.functionSet = functionSet
        self.functionSetLen = len(functionSet)
        self.scalar = config.inputScalarR
        self.totalGenes = config.genomeSize
        self.n = index
        self.fraction = self.n / float(config.genomeSize + config.inputs)
        self.output = 0
        self.mutate()

    def rand(self):
        return uniform(0.0, 1.0)

    def mutate(self):
        self.x = self.rand()
        self.y = self.rand()
        self.f = self.rand()
        self.p = self.rand()

    def randIndex(self, array):
        return randint(0, len(array) - 1)

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
        self.output = 0

    def evaluate(self, x, y, p):
        function = self.get_function()
        self.output = self.constrain_result(p * function(x, y, p))

    def constrain_result(self, result):
        constrainedResult = np.array(result)
        originalShape = constrainedResult.shape
        constrainedResult = constrainedResult.flatten()
        for i in range(constrainedResult.size):
            constrainedResult[i] = self.constrain_bounds(constrainedResult[i])
        return constrainedResult.reshape(originalShape)

    def constrain_bounds(self, e):
        if (e > 1.0):
            return 1.0
        elif e < -1.0:
            return -1.0
        else:
            return e
