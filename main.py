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
FRAMES_TO_SKIP_AT_START = 10


def constrain(eightBitArray):
    return eightBitArray / 255.0


def main():
    env = Environment(TETRIS_FILE_PATH)
    width = env.WIDTH
    height = env.HEIGHT

    print(len(Key))

    '''
    while True:
        env.start_episode()
        for _ in range(FRAMES_TO_SKIP_AT_START):
            env.run_frame()
        while env.is_running():
            pixels = np.array(env.get_pixels())
            pixels = pixels.reshape((height, width))
            rPixels = constrain((pixels >> 24) & 255)
            gPixels = constrain((pixels >> 16) & 255)
            bPixels = constrain((pixels >> 8) & 255)
    '''


    '''
    config = Config()
    functionSet = FunctionSet()
    r = numpy.random.rand(SIZE, SIZE)
    g = numpy.random.rand(SIZE, SIZE)
    b = numpy.random.rand(SIZE, SIZE)
    print('start...')
    genome = Genome(config, functionSet)
    print np.max(genome.evaluate(r, g, b))
    '''

if __name__ == '__main__':
    main()
