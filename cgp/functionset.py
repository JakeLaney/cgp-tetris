
import cgp.functions.mathematics
import cgp.functions.comparison
import cgp.functions.lists
import cgp.functions.statistics

class FunctionSet():
    def __init__(self):
        self.functions = []
        self.functions += cgp.functions.mathematics.FUNCTIONS
        self.functions += cgp.functions.lists.FUNCTIONS
        self.functions += cgp.functions.comparison.FUNCTIONS
        self.functions += cgp.functions.statistics.FUNCTIONS
        print('Function set size: ', len(self.functions))

    def __getitem__(self, index):
        return self.functions[index]

    def __len__(self):
        return len(self.functions)
