
class Config():
    # The number of input genes
    inputs = 3

    # The number of output genes
    outputs = 10

    # The number of function genes
    function_genes = 40

    # The size of the genome
    genome_size = self.inputs + self.function_genes

    # evolution hyperparameters
    input_scalar_r = 0.1
    children_per_generations = 9
    genes_mutated = 0.1
    outputs_mutated = 0.6
    generations = 10
