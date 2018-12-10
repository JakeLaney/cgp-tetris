
#import gym
from tetris_learning_environment.gym import TetrisEnvironment

env = TetrisEnvironment('../Tetris.gb')
while True:
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(0)
