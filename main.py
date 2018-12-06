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

from cgp import functional_graph

FRAMES_TO_SKIP_AT_START = 75
FRAMES_TO_SKIP_EACH_TURN = 60
NUM_KEYS = 5
INDIVIDUALS = 1000

def constrain(eightBitArray):
    return eightBitArray / 255.0

def worker_init(rom_path):
    global env
    global function_set
    global config
    env = Environment(rom_path)
    function_set = FunctionSet()
    config = Config()
    config.inputs = 3
    config.outputs = NUM_KEYS * 2  # because we have booleans
    config.functionGenes = 40


def play_game(genome):
    key_presses = [False] * int(config.outputs / 2)
    env.start_episode()
    for _ in range(FRAMES_TO_SKIP_AT_START):
        env.run_frame()

    while env.is_running():
        pixels = np.array(env.get_pixels())
        pixels = pixels.reshape((env.HEIGHT, env.WIDTH))
        rPixels = constrain((pixels >> 24) & 255)
        gPixels = constrain((pixels >> 16) & 255)
        bPixels = constrain((pixels >> 8) & 255)

        output = genome.evaluate(rPixels, gPixels, bPixels)

        for i in range(NUM_KEYS):
            shouldPressKey = output[i] > output[i + 1]
            env.set_key_state(i + 1, shouldPressKey)
            key_presses[i] = shouldPressKey
        for _ in range(FRAMES_TO_SKIP_EACH_TURN):
            env.run_frame()

    score = env.get_score()
    return (genome, score)


def main():
    if len(sys.argv) < 2:
        print("Missing rom path argument.")
        return

    tetris_rom_path = sys.argv[1]

    functionSet = FunctionSet()

    config = Config()
    config.inputs = 3
    config.outputs = NUM_KEYS * 2 # because we have booleans
    config.functionGenes = 40
    config.generations = int(INDIVIDUALS / config.childrenPerGeneration)

    bestScore = 0
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

    graph = functional_graph.FunctionalGraph(elite)
    graph.draw(0)

if __name__ == '__main__':
    main()
