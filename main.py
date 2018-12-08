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

from cgp import functional_graph

import signal
import time

FRAME_SKIP = 120
DOWNSAMPLE = 8
PROCESSES = 3

CONFIG = Config()
FUNCTION_SET = FunctionSet()

def worker_init(rom_path):
    global env
    env = gym.TetrisEnvironment(rom_path, frame_skip=FRAME_SKIP)

def run_episode(genome):
    pixels = env.reset()
    done = False
    rewardSum = 0
    while not done:
        grayscale = np.sum(pixels, axis = 2) / 3.0 / 255.0 # constrained to range [0, 1]
        #rPixels = pixels[::DOWNSAMPLE,::DOWNSAMPLE,0] +
        #gPixels = pixels[::DOWNSAMPLE,::DOWNSAMPLE,1] / 255.0
        #bPixels = pixels[::DOWNSAMPLE,::DOWNSAMPLE,2] / 255.0
        output = genome.evaluate(grayscale)
        action = np.argmax(output)
        pixels, reward, done, info = env.step(action)
        rewardSum += reward + 1
    return (genome, rewardSum)

def render(env, genome):
    pixels = env.reset()
    import pygame
    pygame.init()
    size = (pixels.shape[1], pixels.shape[0])
    display = pygame.display.set_mode(size)
    pygame.display.set_caption('Tetris')
    carryOn = True
    clock = pygame.time.Clock()
    done = False
    while not done and carryOn:
        for event in pygame.event.get(): # User did something
            if event.type == pygame.QUIT: # If user clicked close
                carryOn = False
        pygame.surfarray.blit_array(display, np.flip(np.rot90(pixels), axis=0))
        pygame.display.flip()
        rPixels = pixels[::DOWNSAMPLE,::DOWNSAMPLE,0] / 255.0
        gPixels = pixels[::DOWNSAMPLE,::DOWNSAMPLE,1] / 255.0
        bPixels = pixels[::DOWNSAMPLE,::DOWNSAMPLE,2] / 255.0
        output = genome.evaluate(rPixels, gPixels, bPixels)
        action = np.argmax(output)
        pixels, reward, done, info = env.step(action)
        clock.tick(60)
    pygame.quit()


def main():
    if len(sys.argv) < 2:
        print("Missing rom path argument.")
        return

    tetris_rom_path = sys.argv[1]

    bestScore = 0

    global elite
    elite = Genome(CONFIG, FUNCTION_SET)

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

    env = gym.TetrisEnvironment(tetris_rom_path, frame_skip=FRAME_SKIP)
    while True:
        render(env, elite)

if __name__ == '__main__':
    main()
