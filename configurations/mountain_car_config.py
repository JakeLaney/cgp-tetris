from cgp.functionset import FunctionSet

class MountainCarConfig():
    # elite mode file
    modelFile = 'mountain_car.out'

    functionSet = FunctionSet()

    # The number of input genes
    inputs = 2

    # The number of output genes
    outputs = 3

    # The number of function genes
    functionGenes = 40

    # evolution hyperparameters
    inputScalarR = 0.1
    genesMutated = 0.1
    outputsMutated = 0.6

    individuals = 10000
    childrenPerGeneration = 4
    generations = int(individuals / childrenPerGeneration)

    def get_genome_size(self):
        return self.inputs + self.functionGenes
