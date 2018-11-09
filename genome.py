
import config
import random

from gene import Gene

class Genome:
    def __init__(self):
        self.out = 0
        self.len = config.INPUTS + config.GENOME_SIZE

        self.init_genes()
        self.init_outputs()
    
    def init_genes(self):
        self.genes = []
        for i in xrange(config.INPUTS):
            self.genes.append(Gene(i))
        for i in xrange(config.GENOME_SIZE):
            self.genes.append(Gene(i + config.INPUTS))

    def random_output_index(self):
        return int(round(random.random() * (self.len -1)))
    
    def init_outputs(self):
        self.outputs = []
        for i in xrange(config.OUTPUTS):
            self.outputs.append(self.random_output_index())

    def compute(self, *argv):
        for inputIdx, arg in enumerate(argv):
            self.genes[inputIdx].output = arg

        for i in xrange(config.GENOME_SIZE):
            idx = i + 3
            gene = self.genes[idx]
            f = gene.get_function()
            xGeneOut = self.genes[gene.get_x()].output
            yGeneOut = self.genes[gene.get_y()].output
            p = gene.get_p()
            gene.out = p * f(xGeneOut, yGeneOut, p)

        result = []
        for i in xrange(config.OUTPUTS):
            result.append(self.genes[self.outputs[i]].out)

        return result
        
