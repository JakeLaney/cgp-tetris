#!/usr/local/bin/python3

from config import Config
from cgp.functionset import FunctionSet
from cgp.genome import Genome

import numpy as np
import numpy.random

from PIL import Image
import matplotlib.pyplot as plt

from tetris_learning_environment import Environment
from tetris_learning_environment import Keys

TETRIS_FILE_PATH = '/Users/JakeL/Desktop/Tetris.gb'

def convert(pixel):
    '''convert 32-bit integer to 4 8-bit integers'''
    v = int(pixel)
    r = (v >> 24) & 255
    g = (v >> 16) & 255
    b = (v >> 8) & 255
    a = v & 255
    return (r, g, b, a) #, dtype=np.uint8)

def constrain(eightBitArray):
    return eightBitArray / 255.0

def main():
    env = Environment(TETRIS_FILE_PATH)
    width = env.WIDTH
    height = env.HEIGHT
    env.start_episode()
    for _ in range(100):
        env.run_frame()
    pixels = np.array(env.get_pixels())
    pixels = pixels.shape((width, height))
    rPixels = constrain((pixels >> 24) & 255)
    gPixels = constrain((pixels >> 16) & 255)
    bPixels = constrain((pixels >> 8) & 255)
    aPixels = constrain((pixels >> 0) & 255)
    m = np.stack([rPixels, gPixels, bPixels, aPixels], axis=-1)
    print(m.shape)
    '''
    config = Config()
    functionSet = FunctionSet()
    r = numpy.random.rand(SIZE, SIZE)
    g = numpy.random.rand(SIZE, SIZE)
    b = numpy.random.rand(SIZE, SIZE)
    print 'start...'
    genome = Genome(config, functionSet)
    print np.max(genome.evaluate(r, g, b))
    '''

if __name__ == '__main__':
    main()
