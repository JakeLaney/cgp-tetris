#!/usr/local/bin/python3

import sys
from multiprocessing import Pool

from timeit import default_timer as timer

from configurations import tetris_config
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

CONFIG = Config()
FUNCTION_SET = FunctionSet()

GAME_MARGIN_X = 16
GAME_HEIGHT = 144
GAME_WIDTH = 80

GRID_HEIGHT = 18
GRID_WIDTH = 10

def get_grid(pixels):
    grid = np.mean(pixels, axis=2)
    grid = grid[0:GAME_HEIGHT:8, GAME_MARGIN_X:GAME_MARGIN_X+GAME_WIDTH:8]
    result = np.zeros(grid.shape)
    result[grid > 200] = 0.0
    result[grid <= 200] = 1.0
    return result

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
    last = 0
    actions = [0] * 6
    rewardSum = 0
    while not done and carryOn:
        for event in pygame.event.get(): # User did something
            if event.type == pygame.QUIT: # If user clicked close
                carryOn = False
        pygame.surfarray.blit_array(display, np.flip(np.rot90(pixels), axis=0))
        pygame.display.flip()
        next = estimate_value(pixels)
        reward = last - next
        last = next
        rewardSum += reward
        #print('reward', reward)
        #rPixels = pixels[::8,::8,0] / 255.0
        #gPixels = pixels[::8,::8,1] / 255.0
        #bPixels = pixels[::8,::8,2] / 255.0
        output = genome.evaluate(get_grid(pixels))
        action = np.argmax(output)
        #act = randint(0,5)
        actions[action] += 1
        print(estimate_value(pixels))
        pixels, reward, done, info = env.step(action)
        clock.tick(60)
    print(actions)
    print('reward sum', rewardSum)
    pygame.quit()

def graph(elite):
    from cgp.functional_graph import FunctionalGraph
    graph = FunctionalGraph(elite)
    for i in range(3):
        graph.draw(0)

def main():
    if len(sys.argv) < 2:
        print("Missing rom path argument.")
        return

    tetris_rom_path = sys.argv[1]
    env = gym.TetrisEnvironment(tetris_rom_path, frame_skip=FRAME_SKIP)

    elite = Genome(CONFIG, FUNCTION_SET)
    elite.load_from_file(sys.argv[2])

    while True:
        render(env, elite)

if __name__ == '__main__':
    main()
