
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
        self.len = config.get_genome_size()
        self.config = config
        self.functionSet = functionSet
        self.genes = self.init_genes(config, functionSet)
        self.outputs = Outputs(config)

    def init_genes(self, config, functionSet):
        genes = []
        for i in range(self.len):
            genes.append(Gene(config, functionSet, i))
        return genes

    def get_references(self):
        for gene in self.genes:
            gene.active = False
        ref = [0] * self.len
        for i in self.outputs:
            self.ref_recursive(i, ref)
        return ref

    def ref_recursive(self, index, ref):
        gene = self.genes[index]
        ref[gene.n] + 1
        if not gene.active:
            gene.active = True
            ref[gene.get_x()] += 1
            ref[gene.get_y()] += 1
            self.ref_recursive(gene.get_x(), ref)
            self.ref_recursive(gene.get_y(), ref)

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

        self.mark_active_genes()

        for i in self.functionGeneRange():
            gene = self.genes[i]
            if gene.active:
                xOutput = self.genes[gene.get_x()].output
                yOutput = self.genes[gene.get_y()].output
                pOutput = gene.get_p()
                gene.evaluate(xOutput, yOutput, pOutput)

        result = []
        for outputIndex in self.outputs:
            output = self.genes[outputIndex].output
            result.append(np.mean(output))

        return result

    def mark_active_genes(self):
        for index in self.outputs:
            self.mark_active_genes_recursive(index)

    def mark_active_genes_recursive(self, index):
        gene = self.genes[index]
        if not gene.active:
            gene.active = True
            self.mark_active_genes_recursive(gene.get_x())
            self.mark_active_genes_recursive(gene.get_y())

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
            outputFile.write(str(len(self.genes)))
            outputFile.write('\n')
            outputFile.write(str(len(self.outputs)))
            outputFile.write('\n')
            for gene in self.genes:
                outputFile.write(str(gene.x))
                outputFile.write(',')
                outputFile.write(str(gene.y))
                outputFile.write(',')
                outputFile.write(str(gene.f))
                outputFile.write(',')
                outputFile.write(str(gene.p))
                outputFile.write('\n')
            for output in self.outputs:
                outputFile.write(str(output))
                outputFile.write('\n')

    def load_from_file(self, path):
        with open(path, 'r') as inputFile:
            numGenes = int(inputFile.readline())
            numOutputs = int(inputFile.readline())

            genes = []
            outputs = []

            for i in range(numGenes):
                values = inputFile.readline().split(',')
                gene = Gene(self.config, self.functionSet, i)
                gene.x = float(values[0])
                gene.y = float(values[1])
                gene.f = float(values[2])
                gene.p = float(values[3])
                genes.append(gene)

            for _ in range(numOutputs):
                outputs.append(int(inputFile.readline()))

            self.genes = genes
            self.outputs = Outputs(self.config)
            self.outputs.load_from_list(outputs)
