
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from carle import AutomaticCellularEnvironment
from mcl import RND2D 

import matplotlib.pyplot as plt


class RandomNetworkAgent(nn.Module):

    def __init__(self, **kwargs):
        super(RandomNetworkAgent, self).__init__()

        self.action_width = kwargs["action_width"] \
                if "action_width" in kwargs.keys()\
                else 32
        self.action_height = kwargs["action_height"] \
                if "action_height" in kwargs.keys()\
                else 32
        self.observation_width = kwargs["observation_width"] \
                if "action_width" in kwargs.keys()\
                else 64
        self.observation_height = kwargs["observation_height"] \
                if "action_height" in kwargs.keys()\
                else 64

        self.depth = 3
        self.filter_dim = 4



        self.initialize_network()

    def initialize_network(self):

        dense_nodes = (self.observation_width // 4) * (self.observation_height // 4)
        output_nodes = self.action_width * self.action_height

        self.network = nn.Sequential(\
                nn.Conv2d(1, self.filter_dim, 3, padding=1, bias=False),\
                nn.ReLU(),\
                nn.MaxPool2d(2, 2, padding=0),\
                nn.Conv2d(self.filter_dim, 1, 3, padding=1, bias=False),\
                nn.ReLU(),\
                nn.MaxPool2d(2, 2, padding=0),\
                nn.Flatten(),\
                nn.Linear(dense_nodes, output_nodes, bias=False),\
                nn.Sigmoid())

        for param in self.network.parameters():

            param.requires_grad = False

    def forward(self, obs):

        instances = obs.shape[0]

        output = self.network(obs)

        action = 1.0 * (output <= torch.rand_like(output))

        action = action.reshape(instances, 1, \
                self.action_width, self.action_height)

        return action


if __name__ == "__main__":
    
    
    #high life: 23/36
    #life: 23/3
    my_steps = 8192

    for rules, name in zip([[[2,3],[3]], [[2,3],[3,6]]], ["life", "high_life"]):

        agent = RandomNetworkAgent() 
        env_fn = AutomaticCellularEnvironment 
        env = RND2D(env_fn)

        env.env.survive = rules[0]
        env.env.birth = rules[1]
        env.env.instances = 1


        obs = env.reset()
        rewards = []
        steps = []

        for step in range(my_steps):

            action = agent(obs)
            obs, reward, done, info = env.step(action)

            rewards.append(reward.squeeze().detach().numpy())
            steps.append(step)

            #env.env.save_frame()

            fig = plt.figure(figsize=(6,12))
            plt.subplot(211)
            plt.plot(steps, rewards)
            plt.plot(steps[-1], rewards[-1], "o")
            plt.title("RND reward in {} CA".format(name))
            plt.subplot(212)
            plt.imshow(obs[0,0,:,:].detach().numpy())
            plt.title("{} CA".format(name))
            plt.savefig("./frames/rewards_{}_step{}".format(name, step))
            plt.close(fig)