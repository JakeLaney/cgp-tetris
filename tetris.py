#!/usr/local/bin/python3

import sys

import configurations.tetris_config
from cgp.training_environment import TrainingEnvironment
from cgp.trainer.tetris_trainer import TetrisTrainer

def main():
    if len(sys.argv) < 2:
        print("Missing rom path argument.")
        return
    if len(sys.argv) == 3:
        modelFile = sys.argv[2]
        shouldLoadFromFile = True

    cgpConfig = configurations.tetris_config.TetrisConfig()
    romPath = sys.argv[1]
    trainer = TetrisTrainer(romPath)
    trainingEnv = TrainingEnvironment()

    print('Training Tetris CGP Model...')

    elite, bestScore = trainingEnv.run(trainer, cgpConfig)

    print('Completed with score:', bestScore)

if __name__ == '__main__':
    main()
