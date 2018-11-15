
class Config():
    # The number of input genes
    inputs = 3

    # The number of output genes
    outputs = 10

    # The number of function genes
    functionGenes = 40

    # The size of the genome
    genomeSize = inputs + functionGenes

    # evolution hyperparameters
    inputScalarR = 0.1
    childrenPerGeneration = 9
    genesMutated = 0.1
    outputsMutated = 0.6
    generations = 10
