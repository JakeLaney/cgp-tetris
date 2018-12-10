from cgp.functionset import FunctionSet
import tetris_learning_environment.gym as gym
import configurations.tetris_heuristic as heuristic

import numpy as np
from random import randint

class HeuristicTrainer:
    FRAME_SKIP = 60

    GAME_MARGIN_X = 16
    GAME_HEIGHT = 144
    GAME_WIDTH = 80
    GRID_HEIGHT = 18
    GRID_WIDTH = 10

    def __init__(self, romPath):
        self.romPath = romPath
        self.lastScore = 0

    def reset(self):
        self.lastScore = 0

    def get_env(self):
        env = gym.TetrisEnvironment(self.romPath, frame_skip=self.FRAME_SKIP)
        return env

    def run_episode_async(self, genome):
        env = self.get_env()
        pixels = env.reset() # ndarray.shape is (_, _, 3) rgb
        sumRewards = 0
        done = False
        actions = [0] * 6
        while not done:
            tetrisGrid = self.downsample(pixels)
            outputVector = genome.evaluate(tetrisGrid)
            selectedAction = np.argmax(outputVector)
            actions[selectedAction] += 1
            pixels, _, done, _ = env.step(selectedAction)
            sumRewards += self.heuristic_reward(pixels, selectedAction)
        print('####', actions, sumRewards)
        return (genome, sumRewards)

    def run_episode(self, env, genome):
        pixels = env.reset() # ndarray.shape is (_, _, 3) rgb
        sumRewards = 0
        done = False
        actions = [0] * 6
        while not done:
            tetrisGrid = self.downsample(pixels)
            outputVector = genome.evaluate(tetrisGrid)
            selectedAction = np.argmax(outputVector)
            actions[selectedAction] += 1
            pixels, _, done, _ = env.step(selectedAction)
            sumRewards += self.heuristic_reward(pixels, selectedAction)
        print('####', actions, sumRewards)
        return (genome, sumRewards)

    def downsample(self, pixels):
        grid = np.mean(pixels, axis=2)
        grid = grid[0:self.GAME_HEIGHT:8, self.GAME_MARGIN_X:self.GAME_MARGIN_X+self.GAME_WIDTH:8]
        result = np.zeros(grid.shape)
        result[grid > 200] = 0.0
        result[grid <= 200] = 1.0
        return result

    def heuristic_reward(self, pixels, action):
        currentScore = heuristic.estimate_value(pixels, debug=False)
        reward = currentScore - self.lastScore
        self.lastScore = currentScore
        return reward

class TetrisConfig():
    def __init__(self, romPath):
        self.heuristicTrainer = HeuristicTrainer(romPath)

    # elite mode file
    modelFile = 'tetris.out'

    functionSet = FunctionSet()

    # The number of input genes
    inputs = 1

    # The number of output genes
    outputs = 6

    # The number of function genes
    functionGenes = 40

    # evolution hyperparameters
    inputScalarR = 0.1
    genesMutated = 0.4
    outputsMutated = 0.6

    individuals = 10000
    childrenPerGeneration = 9
    generations = int(individuals / childrenPerGeneration)

    def get_genome_size(self):
        return self.inputs + self.functionGenes
