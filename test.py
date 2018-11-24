
from config import Config
from cgp.functionset import FunctionSet
from cgp.genome import Genome


functionSet = FunctionSet()
config = Config()
config.inputs = 3
config.outputs = 10 # because we have booleans
config.functionGenes = 40
keyPresses = [False] * int(config.outputs / 2)

elite = Genome(config, functionSet)
elite.save_to_file('jake.out')
