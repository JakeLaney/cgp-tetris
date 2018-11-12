
from random import uniform

class Outputs():
    def __init__(self, config):
        self.len = config.OUTPUTS
        self.genomeSize = config.GENOME_SIZE
        self.init_outputs()

    def init_outputs(self):
        self.outputs = []
        for _ in xrange(self.len):
            self.outputs.append(self.random_genome_index())
    
    def random_genome_index(self):
        randValue = uniform(0.0, 1.0)
        index = int(round(randValue * (self.genomeSize - 1)))
        return index

    def mutate(self):
        pass # TODO 

    def __getitem__(self, index):
        return self.outputs[index]

    def __len__(self):
        return len(self.outputs)

    def __iter__(self):
        return self.outputs.__iter__()

        
