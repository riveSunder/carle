import unittest
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from carle.mcl import MorphoBonus, PredictionBonus, get_glider
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbosity", default=0, \
        help="Verbosity, options are 0 (quiet, default), 1 (timid), and 2 (noisy)")

    args = parser.parse_args()

    unittest.main(verbosity=args.verbosity)
