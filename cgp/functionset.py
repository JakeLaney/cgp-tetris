
import functions.mathematics
import functions.comparison
import functions.lists
import functions.statistics

class FunctionSet():
    def __init__(self):
        self.functions = []
        self.functions += functions.mathematics.FUNCTIONS
        self.functions += functions.lists.FUNCTIONS
        self.functions += functions.comparison.FUNCTIONS
        self.functions += functions.statistics.FUNCTIONS

    def __getitem__(self, index):
        return self.functions[index]

    def __len__(self):
        return len(self.functions)
        
