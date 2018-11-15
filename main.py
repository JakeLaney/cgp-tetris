#!/usr/local/bin/python3

from config import Config
from cgp.functionset import FunctionSet
from cgp.genome import Genome

import numpy as np
import numpy.random

from PIL import Image
import matplotlib.pyplot as plt

from tetris_learning_environment import Environment
from tetris_learning_environment import Key

TETRIS_FILE_PATH = '/Users/JakeL/Desktop/Tetris.gb'
FRAMES_TO_SKIP_AT_START = 75
FRAMES_TO_SKIP_EACH_TURN = 60
NUM_KEYS = 5

def constrain(eightBitArray):
    return eightBitArray / 255.0

def main():
    env = Environment(TETRIS_FILE_PATH)
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
    for _ in range(config.generations):
        children = []
        for _ in range(config.childrenPerGeneration):
            children.append(elite.get_child())

        for childNumber, child in enumerate(children):
            progress = counter / (config.generations * config.childrenPerGeneration)
            print('progress: ', progress)
            env.start_episode()
            for _ in range(FRAMES_TO_SKIP_AT_START):
                env.run_frame()
            steps = 0
            while env.is_running():
                pixels = np.array(env.get_pixels())
                pixels = pixels.reshape((height, width))
                rPixels = constrain((pixels >> 24) & 255)
                gPixels = constrain((pixels >> 16) & 255)
                bPixels = constrain((pixels >> 8) & 255)
                aPixels = constrain((pixels >> 0) & 255)
                output = child.evaluate(rPixels, gPixels, bPixels)
                #m = np.stack([rPixels, gPixels, bPixels, aPixels], axis=-1)
                #print(m.shape)
                #plt.imshow(m)
                #plt.show()
                for i in range(NUM_KEYS):
                    shouldPressKey = output[i] > output[i + 1]
                    env.set_key_state(i + 1, shouldPressKey)
                    keyPresses[i] = shouldPressKey
                for _ in range(FRAMES_TO_SKIP_EACH_TURN):
                    env.run_frame()
            score = env.get_score()
            if score > bestScore:
                bestScore = score
                elite = child
                counter += config.childrenPerGeneration - (childNumber + 1)
                break
            counter += 1
            print('score: ', score)
            print('Best: ', bestScore)
    print("FINISHED")
    print(bestScore)

if __name__ == '__main__':
    main()
