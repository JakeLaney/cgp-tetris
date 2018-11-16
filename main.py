#!/usr/local/bin/python3

import sys
from queue import Queue
from threading import Thread, local
from collections import deque

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


def main():
    if len(sys.argv) < 2:
        print("Missing rom path argument.")
        return

    tetris_rom_path = sys.argv[1]

    work_queue = Queue()  # queue of genomes to evaluate
    result_queue = deque()  # queue of (genome, score) tuples

    def worker():
        env = Environment(tetris_rom_path)
        while True:
            genome = work_queue.get(True, None)

            env.start_episode()
            for _ in range(FRAMES_TO_SKIP_AT_START):
                env.run_frame()

            while env.is_running():
                pixels = np.array(env.get_pixels())
                pixels = pixels.reshape((height, width))
                rPixels = constrain((pixels >> 24) & 255)
                gPixels = constrain((pixels >> 16) & 255)
                bPixels = constrain((pixels >> 8) & 255)

                output = genome.evaluate(rPixels, gPixels, bPixels)

                for i in range(NUM_KEYS):
                    shouldPressKey = output[i] > output[i + 1]
                    env.set_key_state(i + 1, shouldPressKey)
                    keyPresses[i] = shouldPressKey
                for _ in range(FRAMES_TO_SKIP_EACH_TURN):
                    env.run_frame()

            score = env.get_score()
            print("score: ", score)
            result_queue.append((genome, score))

            work_queue.task_done()

    # create some worker threads
    thread_count = 4
    for _ in range(thread_count):
        Thread(target=worker).start()

    env = Environment(tetris_rom_path)
    width = env.WIDTH
    height = env.HEIGHT

    functionSet = FunctionSet()
    config = Config()
    config.inputs = 3
    config.outputs = NUM_KEYS * 2 # because we have booleans
    config.functionGenes = 40
    keyPresses = [False] * int(config.outputs / 2)

    elite = Genome(config, functionSet)
    bestScore = 0
    counter = 0.0

    for generation in range(config.generations):
        for _ in range(config.childrenPerGeneration):
            work_queue.put(elite.get_child(), True)

        # wait for all of the children to be evaluated
        work_queue.join()

        for (genome, score) in result_queue:
            if score > bestScore:
                bestScore = score
                elite = genome

        result_queue.clear()

        print('Generation ' + str(generation) + ' end, current best score = ', bestScore)
    print("FINISHED")
    print(bestScore)


if __name__ == '__main__':
    main()
