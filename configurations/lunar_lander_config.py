from cgp.functionset import FunctionSet

class LunarLanderConfig():
    # elite mode file
    modelFile = 'lunar_lander.out'

    functionSet = FunctionSet()

    # The number of input genes
    inputs = 8

    # The number of output genes
    outputs = 4

    # The number of function genes
    functionGenes = 40

    # evolution hyperparameters
    inputScalarR = 0.1
    genesMutated = 0.1
    outputsMutated = 0.6

    individuals = 1000
    childrenPerGeneration = 4
    generations = int(individuals / childrenPerGeneration)

    def get_genome_size(self):
        return self.inputs + self.functionGenes
