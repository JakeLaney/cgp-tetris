
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

        self.function_descriptions = []
        self.function_descriptions += cgp.functions.mathematics.FUNCTION_NAMES
        self.function_descriptions += cgp.functions.lists.FUNCTION_NAMES
        self.function_descriptions += cgp.functions.comparison.FUNCTION_NAMES
        self.function_descriptions += cgp.functions.statistics.FUNCTION_NAMES

    def __getitem__(self, index):
        return self.functions[index]

    def __len__(self):
        return len(self.functions)
