
import random

from functions import FUNCTIONS

class Gene:
    def __init__(self, col):
        self.col = col
        self.mutate()
    
    def mutate(self):
        self.f = self.randFunction()
        if self.col != 0:
            self.a = self.randCoor()
            self.b = self.randCoor()

    def randIndex(self, array):
        return random.randint(0, len(array) - 1)        
        
    def randFunction(self):
        return FUNCTIONS[self.randIndex(FUNCTIONS)]

    def randCoor(self):
        return random.randint(0, self.col - 1)

    def compute(self, genome, x, y):
        if self.col == 0:
            return x
        elif self.col == 1:
            return y
        else:
            inputX = genome[self.a].compute(genome, x, y)
            inputY = genome[self.b].compute(genome, x, y)
            return self.f(inputX, inputY)

    

