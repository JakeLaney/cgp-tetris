#!/usr/local/bin/python3

import sys

import configurations.lunar_lander_config
from cgp.training_environment import TrainingEnvironment
from cgp.trainer.lunar_lander_v2 import LunarLanderTrainer

def main():
    cgpConfig = configurations.lunar_lander_config.LunarLanderConfig()
    trainer = LunarLanderTrainer()
    trainingEnv = TrainingEnvironment()

    print('Training Mountain Car CGP Model...')

    elite, bestScore = trainingEnv.run(trainer, cgpConfig)

    print('Completed with score:', bestScore)

    import numpy as np
    env = LunarLanderTrainer().get_env()
    while True:
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action = elite.evaluate(obs)
            obs, reward , done, _ = env.step(np.argmax(action))

if __name__ == '__main__':
    main()
