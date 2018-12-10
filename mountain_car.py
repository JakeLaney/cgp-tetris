#!/usr/local/bin/python3

import sys

import configurations.mountain_car_config
from cgp.training_environment import TrainingEnvironment
from cgp.trainer.mountain_car_trainer import MountainCarTrainer

def main():
    cgpConfig = configurations.mountain_car_config.MountainCarConfig()
    trainer = MountainCarTrainer()
    trainingEnv = TrainingEnvironment()

    print('Training Mountain Car CGP Model...')

    elite, bestScore = trainingEnv.run(trainer, cgpConfig)

    print('Completed with score:', bestScore)

if __name__ == '__main__':
    main()
