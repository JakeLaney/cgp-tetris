
class Config():
    # The number of input genes
    inputs = 3

    # The number of output genes
    outputs = 3

    # The number of function genes
    functionGenes = 40

    # evolution hyperparameters
    inputScalarR = 0.2
    genesMutated = 0.5
    outputsMutated = 0.6

    individuals = 1000
    childrenPerGeneration = 9
    generations = int(individuals / childrenPerGeneration)

    def get_genome_size(self):
        return self.inputs + self.functionGenes
