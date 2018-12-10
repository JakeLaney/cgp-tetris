import gym
import numpy as np

class LunarLanderTrainer:
    def bootstrap_process(self):
        pass

    def reset(self):
        pass

    def get_env(self):
        return gym.make('LunarLander-v2')

    def run_episode(self, env, genome):
        observation = env.reset()
        sumRewards = 0
        done = False
        while not done:
            inputs = self.constrain_observation(observation)
            outputVector = genome.evaluate(*inputs)
            selectedAction = np.argmax(outputVector)
            observation, reward, done, _ = env.step(selectedAction)
            sumRewards += reward
        return (genome, sumRewards)

    def constrain_observation(self, observation):
        return observation / 100.0
