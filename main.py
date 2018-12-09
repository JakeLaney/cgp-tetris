#!/usr/local/bin/python3

import sys
from multiprocessing import Pool

from timeit import default_timer as timer

from config import Config
from cgp.functionset import FunctionSet
from cgp.genome import Genome

import numpy as np
import numpy.random

from random import randint

from tetris_learning_environment import Environment
from tetris_learning_environment import Key
import tetris_learning_environment.gym as gym

from heuristic import estimate_value

from cgp import functional_graph

import signal
import time

FRAME_SKIP = 15
PROCESSES = 4

CONFIG = Config()
FUNCTION_SET = FunctionSet()

def worker_init(rom_path):
    global env
    env = gym.TetrisEnvironment(rom_path, frame_skip=FRAME_SKIP)

def downsample(pixels):
    return np.mean(pixels[::8, ::8, :]) / 255.0

def run_episode(genome):
    pixels = env.reset()
    done = False
    rewardSum = 0
    last = 0
    next = 0
    while not done:
        grayscale = downsample(pixels)
        output = genome.evaluate(grayscale)
        action = np.argmax(output)
        pixels, _, done, _ = env.step(action)
        next = estimate_value(pixels)
        reward = next - last
        last = next
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
    if len(sys.argv) == 3:
        elite.load_from_file(sys.argv[2])

    print('Starting CGP for ' + str(CONFIG.generations) + ' generations...')

    with Pool(processes=PROCESSES, initializer=worker_init, initargs=(tetris_rom_path,)) as pool:
        for generation in range(CONFIG.generations):
            start = timer()

            children = [elite.get_child() for _ in range(CONFIG.childrenPerGeneration)]
            results = [pool.apply_async(run_episode, args=(child,)) for child in children]
            results = [result.get() for result in results]

            for (genome, score) in results:
                if score >= bestScore:
                    bestScore = score
                    elite = genome
                    elite.save_to_file('elite.out')

            end = timer()

            timeElapsed = end - start
            estimatedTimeSec = timeElapsed * (CONFIG.generations + 1 - generation)
            estimatedTimeMin = estimatedTimeSec / 60.0

            print('Generation ' + str(generation + 1) + ' of ' + str(CONFIG.generations) + ' complete, current best score = ', bestScore)
            print('Est. minutes remaining: ' + str(estimatedTimeMin))

    print("FINISHED")
    print('Best Score: ', bestScore)

if __name__ == '__main__':
    main()
