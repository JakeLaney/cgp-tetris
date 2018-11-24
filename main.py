#!/usr/local/bin/python3

import sys
from multiprocessing import Pool

from config import Config
from cgp.functionset import FunctionSet
from cgp.genome import Genome

import numpy as np
import numpy.random

from PIL import Image
import matplotlib.pyplot as plt

from tetris_learning_environment import Environment
from tetris_learning_environment import Key

FRAMES_TO_SKIP_AT_START = 75
FRAMES_TO_SKIP_EACH_TURN = 60
NUM_KEYS = 5


def constrain(eightBitArray):
    return eightBitArray / 255.0


#env = None
#config = None
#function_set = None


def worker_init(rom_path: str):
    global env
    global function_set
    global config
    env = Environment(rom_path)
    function_set = FunctionSet()
    config = Config()
    config.inputs = 3
    config.outputs = NUM_KEYS * 2  # because we have booleans
    config.functionGenes = 40


def play_game(genome: Genome):
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
    print("score: ", score)

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
    keyPresses = [False] * int(config.outputs / 2)

    elite = Genome(config, functionSet)
    bestScore = 0

    with Pool(processes=4, initializer=worker_init, initargs=(tetris_rom_path,)) as pool:

        for generation in range(config.generations):
            genomes = [elite.get_child() for _ in range(config.childrenPerGeneration)]

            results = [pool.apply_async(play_game, args=(genome,)) for genome in genomes]
            results = [result.get() for result in results]
            for (genome, score) in results:
                if score > bestScore:
                    bestScore = score
                    elite = genome

            print('Generation ' + str(generation + 1) + ' end, current best score = ', bestScore)
    print("FINISHED")
    print(bestScore)


if __name__ == '__main__':
    main()
