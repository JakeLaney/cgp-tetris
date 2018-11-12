
import config
import random

from gene import Gene
from outputs import Outputs
import numpy as np

class Genome:
    def __init__(self, config, functionSet):
        self.inputCount = config.INPUTS
        self.functionGeneStartIdx  = config.INPUTS
        self.len = config.GENOME_SIZE
        self.init_genes(config, functionSet)
        self.outputs = Outputs(config)
    
    def init_genes(self, config, functionSet):
        self.genes = []
        for i in xrange(self.len):
            self.genes.append(Gene(config, functionSet, i))

    def evaluate(self, *inputValues):
        self.prepare_input_genes(inputValues)
        outputList = self.evaluate_function_genes()
        return outputList

    def prepare_input_genes(self, inputValues):
        for inputIdx, input in enumerate(inputValues):
            self.genes[inputIdx].init_as_input_gene(input)

    def evaluate_function_genes(self):
        for i in self.functionGeneXRange():
            self.genes[i].prepare_for_evaluation()
        
        for i in self.functionGeneXRange():
            gene = self.genes[i]
            xOutput = self.genes[gene.get_x()].output
            yOutput = self.genes[gene.get_y()].output
            pOutput = gene.get_p()
            gene.evaluate(xOutput, yOutput, pOutput)

        result = []
        for outputIndex in self.outputs:
            output = self.genes[outputIndex].output
            if np.array(output).size == 0:
                output = 0
            result.append(np.mean(output))
        return result

    def functionGeneXRange(self):
        return xrange(self.functionGeneStartIdx, self.len)

        
