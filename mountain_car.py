#!/usr/local/bin/python3

import sys

import configurations.mountain_car_config
from cgp.training_environment import TrainingEnvironment

def main():
    cgpConfig = configurations.mountain_car_config.MountainCarConfig()
    trainingEnv = TrainingEnvironment()

    print('Training Mountain Car CGP Model...')

    elite, bestScore = trainingEnv.run(cgpConfig.trainer, cgpConfig)

    print('Completed with score:', bestScore)

if __name__ == '__main__':
    main()
