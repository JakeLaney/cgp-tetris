#!/usr/local/bin/python3
import sys
from os import getcwd
sys.path.append(getcwd()) # if run from root
sys.path.append(getcwd() + '/..') # if run from test/

import numpy as np
import time
import pygame

import tetris_learning_environment.gym as gym
from configurations.tetris_config import TetrisConfig
from configurations.tetris_config import HeuristicTrainer
from cgp.genome import Genome

def get_pygame_display(pixels):
    pygame.init()
    size = (pixels.shape[1], pixels.shape[0])
    display = pygame.display.set_mode(size)
    pygame.display.set_caption('Tetris')
    return display

def show(display, pixels):
    pygame.surfarray.blit_array(display, np.flip(np.rot90(pixels), axis=0))
    pygame.display.flip()

def game_loop(env, genome):
    pixels = env.reset()
    display = get_pygame_display(pixels)
    actions = [0] * 6
    rewardSum = 0
    done = False
    clock = pygame.time.Clock()
    trainer = HeuristicTrainer(None)
    while not done:
        show(display, pixels)

        outputVector = genome.evaluate(trainer.downsample(pixels))
        selectedAction = np.argmax(outputVector)
        actions[selectedAction] += 1
        pixels, score, done, _ = env.step(selectedAction)
        rewardSum += trainer.heuristic_reward(pixels, selectedAction)

        for event in pygame.event.get(): # User did something
            if event.type == pygame.QUIT: # If user clicked close
                done = True
        clock.tick(60)
    print('reward sum', rewardSum)
    pygame.quit()


def graph(elite):
    from cgp.functional_graph import FunctionalGraph
    graph = FunctionalGraph(elite)
    for i in range(3):
        graph.draw(0)

def main():
    if len(sys.argv) < 3:
        print("usage: python run_tetris.py <rom_path> <genome_path>")
        return

    romPath = sys.argv[1]
    config = TetrisConfig(romPath)
    env = gym.TetrisEnvironment(romPath, frame_skip=config.heuristicTrainer.FRAME_SKIP)

    elite = Genome(config)
    elite.load_from_file(sys.argv[2])

    while True:
        game_loop(env, elite)

if __name__ == '__main__':
    main()
