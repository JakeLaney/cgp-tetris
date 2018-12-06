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

from tetris_learning_environment import Environment
from tetris_learning_environment import Key
import tetris_learning_environment.gym as gym

from cgp import functional_graph

import signal
import time

FRAME_SKIP = 60
INDIVIDUALS = 1000

def worker_init(rom_path):
    global env
    global function_set
    global config
    env = gym.TetrisEnvironment(rom_path, frame_skip=FRAME_SKIP, reward_type=gym.Metric.LINES)
    function_set = FunctionSet()
    config = Config()
    config.inputs = 3
    config.outputs = len(gym.Action) # because we have booleans
    config.functionGenes = 40

def play_game(genome):
    pixels = env.reset()
    done = False
    rewardSum = 0
    while not done:
        rPixels = pixels[:,:,0] / 255.0
        gPixels = pixels[:,:,1] / 255.0
        bPixels = pixels[:,:,2] / 255.0
        output = genome.evaluate(rPixels, gPixels, bPixels)
        action = np.argmax(output)
        pixels, reward, done, info = env.step(action)
        rewardSum += reward

    return (genome, rewardSum)


def main():
    if len(sys.argv) < 2:
        print("Missing rom path argument.")
        return

    tetris_rom_path = sys.argv[1]

    functionSet = FunctionSet()

    config = Config()
    config.inputs = 3
    config.outputs = len(gym.Action) # because we have booleans
    config.functionGenes = 40
    config.generations = int(INDIVIDUALS / config.childrenPerGeneration)

    bestScore = 0

    global elite

    elite = Genome(config, functionSet)

    print('Starting CGP for ' + str(config.generations) + ' generations...')

    with Pool(processes=4, initializer=worker_init, initargs=(tetris_rom_path,)) as pool:

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
