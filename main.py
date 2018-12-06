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

FRAME_SKIP = 1000
INDIVIDUALS = 1000
DOWNSAMPLE = 8

CONFIG = Config()
CONFIG.inputs = 3
CONFIG.outputs = len(gym.Action) # because we have booleans
CONFIG.functionGenes = 40
CONFIG.childrenPerGeneration = 4
CONFIG.generations = int(INDIVIDUALS / CONFIG.childrenPerGeneration)

FUNCTION_SET = FunctionSet()

def worker_init(rom_path):
    global env
    env = gym.TetrisEnvironment(rom_path, frame_skip=FRAME_SKIP)

def play_game(genome):
    pixels = env.reset()
    done = False
    rewardSum = 0
    while not done:
        rPixels = pixels[::DOWNSAMPLE,::DOWNSAMPLE,0] / 255.0
        gPixels = pixels[::DOWNSAMPLE,::DOWNSAMPLE,1] / 255.0
        bPixels = pixels[::DOWNSAMPLE,::DOWNSAMPLE,2] / 255.0
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

    bestScore = 0

    global elite
    elite = Genome(CONFIG, FUNCTION_SET)

    print('Starting CGP for ' + str(CONFIG.generations) + ' generations...')

    with Pool(processes=4, initializer=worker_init, initargs=(tetris_rom_path,)) as pool:
        for generation in range(CONFIG.generations):
            start = timer()

            children = [elite.get_child() for _ in range(CONFIG.childrenPerGeneration)]
            results = [pool.apply_async(play_game, args=(child,)) for child in children]
            results = [result.get() for result in results]

            for (genome, score) in results:
                if score > bestScore:
                    bestScore = score
                    elite = genome

            end = timer()

            timeElapsed = end - start
            estimatedTimeSec = timeElapsed * (CONFIG.generations + 1 - generation)
            estimatedTimeMin = estimatedTimeSec / 60.0

            print('Generation ' + str(generation + 1) + ' of ' + str(CONFIG.generations) + ' complete, current best score = ', bestScore)
            print('Est. minutes remaining: ' + str(estimatedTimeMin))

    print("FINISHED")
    print('Best Score: ', bestScore)

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
