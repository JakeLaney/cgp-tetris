
class Config():
    # The number of input genes
    INPUTS = 3

    # The number of output genes
    OUTPUTS = 32

    # The number of function genes
    FUNCTION_GENES = 40

    # The size of the genome
    GENOME_SIZE = INPUTS + FUNCTION_GENES

    # evolution hyperparameters
    INPUT_SCALAR_R = 0.1
    CHILDREN_PER_GENERATION = 9
    GENES_MUTATED_FRACTION = 0.1
    OUTPUTS_MUTATED_FRACTION = 0.6
    GENERATIONS = 10 

