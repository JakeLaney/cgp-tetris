
import random

import config
from functions import FUNCTIONS
from functions import FUNCTIONS_LEN
from math import floor

class Gene:
    def __init__(self, n):
        self.n = n
        self.fraction = self.n / float(config.GENOME_SIZE + config.INPUTS)
        self.output = 0
        self.mutate()
    
    def init_rand(self):
        return random.random()
    
    def mutate(self):
        self.x = self.init_rand()
        self.y = self.init_rand()
        self.f = self.init_rand()
        self.p = self.init_rand()

    def randIndex(self, array):
        return random.randint(0, len(array) - 1)   

    def get_x(self):
        return int(floor(self.x * ((1- self.fraction) * config.INPUT_SCALAR_R + self.fraction)))

    def get_y(self):   
        return int(floor(self.x * ((1 - self.fraction) * config.INPUT_SCALAR_R + self.fraction)))
        
    def get_function(self):
        index = int(round(self.f * (FUNCTIONS_LEN - 1)))
        return FUNCTIONS[index]
    
    def get_p(self):
        return 2.0 * self.p - 1



    

