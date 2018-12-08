
from random import uniform
from random import random

class Outputs():
    def __init__(self, config):
        self.len = config.outputs
        self.genomeSize = config.get_genome_size()
        self.init_outputs()

    def init_outputs(self):
        self.outputs = []
        for _ in range(self.len):
            self.outputs.append(self.random_genome_index())

    def load_from_list(self, outputList):
        self.outputs = outputList
        self.len = len(outputList)

    def random_genome_index(self):
        randValue = uniform(0.0, 1.0)
        index = int(round(randValue * (self.genomeSize - 1)))
        return index

    def mutate(self):
        for i in range(self.len):
            if random() < 0.6:
                self.outputs[i] = self.random_genome_index()

    def __getitem__(self, index):
        return self.outputs[index]

    def __len__(self):
        return len(self.outputs)

    def __iter__(self):
        return self.outputs.__iter__()
