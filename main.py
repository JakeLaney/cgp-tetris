#!/usr/local/bin/python3

import sys
from multiprocessing import Pool

from timeit import default_timer as timer

from config import Config
from cgp.functionset import FunctionSet
from cgp.genome import Genome

import numpy as np
import numpy.random

from PIL import Image
import matplotlib.pyplot as plt

# from tetris_learning_environment import Environment
# from tetris_learning_environment import Key
# import tetris_learning_environment.gym as gym

from cgp import functional_graph

import signal
import time

import gym

INDIVIDUALS = 100

def worker_init():
    global env
    global function_set
    global config
    env = gym.make('MountainCar-v0')

def play_game(genome):
    observation = env.reset()
    done = False
    rewardSum = 0
    while not done:
        output = genome.evaluate(observation[0] / 1.2, observation[1] / 1.2)
        action = np.argmax(output)
        observation, reward, done, info = env.step(action)
        rewardSum += reward
    return (genome, rewardSum)

def run(env, genome):
    observation = env.reset()
    done = False
    rewardSum = 0
    while not done:
        env.render()
        output = genome.evaluate(observation[0] / 1.2, observation[1] / 1.2)
        action = np.argmax(output)
        observation, reward, done, info = env.step(action)
        rewardSum += reward
    return (genome, rewardSum)


def main():
    functionSet = FunctionSet()

    config = Config()
    config.inputs = 2
    config.outputs = 3 # because we have booleans
    config.generations = int(INDIVIDUALS / config.childrenPerGeneration)

    bestScore = -200

    global elite

    elite = Genome(config, functionSet)

    print('Starting CGP for ' + str(config.generations) + ' generations...')

    with Pool(processes=4, initializer=worker_init, initargs=()) as pool:

        for generation in range(config.generations):
            start = timer()

            genomes = [elite.get_child() for _ in range(config.childrenPerGeneration)]

            results = [pool.apply_async(play_game, args=(genome,)) for genome in genomes]
            results = [result.get() for result in results]

            for (genome, score) in results:
                if score > bestScore:
                    bestScore = score
                    elite = genome

            end = timer()
            timeElapsed = end - start
            estimatedTimeSec = timeElapsed * (config.generations + 1 - generation)
            estimatedTimeMin = estimatedTimeSec / 60.0

            print('Generation ' + str(generation + 1) + ' of ' + str(config.generations) + ' complete, current best score = ', bestScore)
            print('Est. minutes remaining: ' + str(estimatedTimeMin))

    print("FINISHED")
    print(bestScore)

    env = gym.make('MountainCar-v0')
    while True:
        print(run(env, elite)[1])

    finish()



def finish():
    if elite is not None:
        elite.save_to_file('elite.out')
        # graph = functional_graph.FunctionalGraph(elite)
        # graph.draw(0)
    exit()

def sigint_handler(signum, frame):
    finish()

signal.signal(signal.SIGINT, sigint_handler)

if __name__ == '__main__':
    main()
