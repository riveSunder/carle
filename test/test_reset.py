import unittest
import argparse

import numpy as np
import torch

from carle.env import CARLE

class TestReset(unittest.TestCase):

    def test_reset(self):
        """
        Test CARLE's master toggle functionality, where an agent can reset
        the environment by setting all toggles to toggle (value of 1.0). 
        """

        self.env = CARLE()

        reset_observation = self.env.reset()

        action = torch.ones(self.env.instances, 1,\
            self.env.action_height, self.env.action_width)

        toggle_observation = self.env.step(action)[0]
        
        action[:,:,0:10,0:10] = 0.0

        normal_observation = self.env.step(action)[0]


        self.assertEqual(toggle_observation.mean().item(), 0.0)
        self.assertEqual(reset_observation.mean().item(), 0.0)

        self.assertEqual(1.0, \
            (1.0 * (reset_observation == toggle_observation)).mean().item())

        self.assertNotEqual(1.0, \
            (1.0 * (toggle_observation == normal_observation)).mean().item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbosity", default=0, \
        help="Verbosity, options are 0 (quiet, default), 1 (timid), and 2 (noisy)")

    args = parser.parse_args()

    unittest.main(verbosity=args.verbosity)

