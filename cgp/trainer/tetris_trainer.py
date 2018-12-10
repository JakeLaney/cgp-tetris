import tetris_learning_environment.gym as gym
import cgp.trainer.tetris_heuristic as heuristic

import numpy as np

class TetrisTrainer:
    FRAME_SKIP = 60

    GAME_MARGIN_X = 16
    GAME_HEIGHT = 144
    GAME_WIDTH = 80
    GRID_HEIGHT = 18
    GRID_WIDTH = 10

    def __init__(self, romPath):
        self.romPath = romPath
        self.lastScore = 0

    def reset(self):
        self.lastScore = 0

    def get_env(self):
        env = gym.TetrisEnvironment(self.romPath, frame_skip=self.FRAME_SKIP)
        return env

    def run_episode(self, env, genome):
        pixels = env.reset() # ndarray.shape is (_, _, 3) rgb
        sumRewards = 0
        done = False
        actions = [0] * 6
        while not done:
            tetrisGrid = self.downsample(pixels)
            outputVector = genome.evaluate(tetrisGrid)
            selectedAction = np.argmax(outputVector)
            pixels, _, done, _ = env.step(0)
            sumRewards += self.heuristic_reward(tetrisGrid, selectedAction)
            actions[selectedAction] += 1
        print('####', actions, sumRewards)
        return (genome, sumRewards)

    def downsample(self, pixels):
        grid = np.mean(pixels, axis=2)
        grid = grid[0:self.GAME_HEIGHT:8, self.GAME_MARGIN_X:self.GAME_MARGIN_X+self.GAME_WIDTH:8]
        result = np.zeros(grid.shape)
        result[grid > 200] = 0.0
        result[grid <= 200] = 1.0
        return result

    def heuristic_reward(self, tetrisGrid, action):
        currentScore = heuristic.estimate_value(tetrisGrid)
        reward = currentScore - self.lastScore
        self.lastScore = currentScore
        if action > 0:
            reward -= 0.01 # try to minimize random movement
        return reward
