#!/usr/local/bin/python3

import sys

import configurations.tetris_config
from cgp.training_environment import TrainingEnvironment

def main():
    if len(sys.argv) < 2:
        print("Missing rom path argument.")
        return
    if len(sys.argv) == 3:
        modelFile = sys.argv[2]
        shouldLoadFromFile = True

    romPath = sys.argv[1]
    cgpConfig = configurations.tetris_config.TetrisConfig(romPath)
    trainingEnv = TrainingEnvironment()

    print('Training Tetris CGP Model...')

    elite, bestScore = trainingEnv.run(cgpConfig.heuristicTrainer, cgpConfig)

    print('Completed with score:', bestScore)

if __name__ == '__main__':
    main()
