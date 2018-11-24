
import config
import random

import copy

from cgp.gene import Gene
from cgp.outputs import Outputs
import numpy as np

class Genome:
    def __init__(self, config, functionSet):
        self.inputCount = config.inputs
        self.functionGeneStartIdx  = config.inputs
        self.len = config.genomeSize
        self.config = config
        self.functionSet = functionSet
        self.init_genes(config, functionSet)
        self.outputs = Outputs(config)

    def init_genes(self, config, functionSet):
        self.genes = []
        for i in range(self.len):
            self.genes.append(Gene(config, functionSet, i))

    def evaluate(self, *inputValues):
        self.prepare_input_genes(inputValues)
        outputList = self.evaluate_function_genes()
        return outputList

    def prepare_input_genes(self, inputValues):
        for inputIdx, input in enumerate(inputValues):
            self.genes[inputIdx].init_as_input_gene(input)

    def evaluate_function_genes(self):

        for i in self.functionGeneRange():
            self.genes[i].prepare_for_evaluation()

        for i in self.functionGeneRange():
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

    def functionGeneRange(self):
        return range(self.functionGeneStartIdx, self.len)

    def get_child(self):
        child = copy.deepcopy(self)
        self.mutate_four_nodes(child)
        child.outputs.mutate()
        return child

    def mutate_four_nodes(self, child):
        for gene in random.sample(child.genes, 4):
            gene.mutate()

    def save_to_file(self, path):
        with open(path, 'w') as outputFile:
            for gene in self.genes:
                outputFile.write(str(gene.x))
                outputFile.write(',')
                outputFile.write(str(gene.y))
                outputFile.write(',')
                outputFile.write(str(gene.f))
                outputFile.write(',')
                outputFile.write(str(gene.p))
                outputFile.write('\n')

    def load_from_file(self, path):
        with open(path, 'r') as inputFile:
            self.genes = []
            index = 0
            for row in inputFile:
                values = row.split(',')
                gene = Gene(self.config, self.functionSet, index)
                gene.x = values[0]
                gene.y = values[1]
                gene.f = values[2]
                gene.p = values[3]
                self.genes.append(gene)
                index += 1