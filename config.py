
class Config():
    # The number of input genes
    inputs = 1

    # The number of output genes
    outputs = 6

    # The number of function genes
    functionGenes = 40

    # evolution hyperparameters
    inputScalarR = 0.2
    genesMutated = 0.4
    outputsMutated = 0.6

    individuals = 10000
    childrenPerGeneration = 4
    generations = int(individuals / childrenPerGeneration)

    def get_genome_size(self):
        return self.inputs + self.functionGenes
