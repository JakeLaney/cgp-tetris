
from math import floor
from random import uniform
from random import randint
import numpy as np

class Gene:
    def __init__(self, config, functionSet, index):
        self.functionSet = functionSet
        self.functionSetLen = len(functionSet)
        self.scalar = config.INPUT_SCALAR_R
        self.totalGenes = config.GENOME_SIZE + config.INPUTS
        self.n = index
        self.fraction = self.n / float(config.GENOME_SIZE + config.INPUTS)
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
        return int(floor(self.x * ((1 - fraction) * self.scalar + fraction)))

    def get_y(self):   
        fraction = self.n / float(self.totalGenes)
        return int(floor(self.x * ((1 - fraction) * self.scalar + fraction)))
    
    def get_p(self):
        return (2.0 * self.p) - 1
        
    def get_function(self):
        index = int(round(self.f * (self.functionSetLen - 1)))
        return self.functionSet[index]

    def init_as_input_gene(self, inputValue):
        self.output = inputValue

    def prepare_for_evaluation(self):
        self.output = 0

    def evaluate(self, x, y, p):
        function = self.get_function()
        self.output = self.constrain_result(p * function(x, y, p))

    def constrain_result(self, result):
        arr = np.array(result)
        result = np.array(map(self.constrain_bounds, arr.flatten()))
        return result.reshape(arr.shape)

    def constrain_bounds(self, e):
        if (e > 1.0):
            return 1.0
        elif e < -1.0:
            return -1.0
        else:
            return e
        
    
   



    

