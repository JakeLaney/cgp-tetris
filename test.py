
from config import Config
from cgp.functionset import FunctionSet
from cgp.genome import Genome
from math import floor


functionSet = FunctionSet()
config = Config()
config.inputs = 3
config.outputs = 10 # because we have booleans
config.functionGenes = 40
keyPresses = [False] * int(config.outputs / 2)

genome = Genome(config, functionSet)
genome.save_to_file('jake.out')
next = Genome(config, functionSet)
next.load_from_file('jake.out')
print(genome.evaluate(1,2,3))
