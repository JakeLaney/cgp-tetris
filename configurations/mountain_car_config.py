from cgp.functionset import FunctionSet

import gym
import numpy as np

class BasicTrainer:
    def bootstrap_process(self):
        pass

    def reset(self):
        pass

    def get_env(self):
        return gym.make('MountainCar-v0')

    def run_episode(self, env, genome):
        observation = env.reset()
        sumRewards = 0
        done = False
        while not done:
            inputs = self.constrain_observation(observation)
            outputVector = genome.evaluate(inputs[0], inputs[1])
            selectedAction = np.argmax(outputVector)
            observation, reward, done, _ = env.step(selectedAction)
            sumRewards += reward
        return (genome, sumRewards)

    def constrain_observation(self, observation):
        return observation / 1.2


class MountainCarConfig():
    trainer = BasicTrainer()

    # elite mode file
    modelFile = 'mountain_car.out'

    functionSet = FunctionSet()

    # The number of input genes
    inputs = 2

    # The number of output genes
    outputs = 3

    # The number of function genes
    functionGenes = 40

    # evolution hyperparameters
    inputScalarR = 0.1
    genesMutated = 0.1
    outputsMutated = 0.6

    individuals = 10000
    childrenPerGeneration = 4
    generations = int(individuals / childrenPerGeneration)

    def get_genome_size(self):
        return self.inputs + self.functionGenes
