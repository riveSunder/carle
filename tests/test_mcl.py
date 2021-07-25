import unittest

import numpy as np
import torch
import matplotlib.pyplot as plt

from carle.mcl import MorphoBonus, PredictionBonus, ParsimonyBonus, get_glider
from carle.env import CARLE

class TestPredictionBonus(unittest.TestCase):

    def setUp(self):

        np.random.seed(42)
        torch.random.manual_seed(42)

    def test_practical(self):


        number_steps = 16

        env = CARLE(device="cpu")# , height=64, widht=64)

        env = PredictionBonus(env) 
        env.batch_size = 2
        obs = env.reset()

        action = get_glider() 

        rewards = []
        obs, initial_reward, done, info = env.step(action)

        for ii in range(2):
            obs, reward, done, info = env.step( action*(ii % 2))
            rewards.append(reward.detach().cpu().numpy().mean())

        obs, second_reward, done, info = env.step(action*0)
        
        action = action * 0.
        rewards.append(initial_reward.detach().cpu().numpy().mean())

        for step in range(number_steps):
            
            obs, reward, done, info = env.step(action)
            rewards.append(reward.detach().cpu().numpy().mean())

        # after learning to predict the oscillator pattern, reward should be higher
        print(f"prediction reward start: {initial_reward.mean():.3}, " \
            f"after few steps {second_reward.mean():.3} " \
            f"after learning to predict {reward.mean():.3}")

        self.assertGreater(reward, initial_reward)
        self.assertGreater(reward, second_reward)

class TestParsimonyBonus(unittest.TestCase):

    def setUp(self):

        np.random.seed(42)
        torch.random.manual_seed(42)


    def test_practical(self):

        env = CARLE(device="cpu")

        # apply reward wrapper(s)

        env = PredictionBonus(env)
        env.batch_size = 2
        number_steps = 16

        env = ParsimonyBonus(env)

        action = get_glider() 

        rewards = []
        obs = env.reset()
        obs, initial_reward, done, info = env.step(action)


        action = torch.zeros(1, 1, env.action_height, env.action_width)

        for step in range(number_steps):
            
            obs, reward, done, info = env.step(action)

        rewards.append(reward.detach().cpu().numpy().mean())

        action[:, :, :env.action_height//2, :] = 1.0


        obs, reward, done, info = env.step(action)

        rewards.append(reward.detach().cpu().numpy().mean())

        print(f"ratio with 0 toggles : ~2048 toggles = {rewards[0]/rewards[-1]}")
        
        # should be about 20 times less 
        self.assertLess(abs(rewards[-1]), abs(rewards[0])/10)



if __name__ == "__main__":

    unittest.main(verbosity=2)
