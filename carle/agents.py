
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from carle import AutomaticCellularEnvironment
from mcl import RND2D 

import matplotlib.pyplot as plt


class RandomAgent(nn.Module):

    def __init__(self, **kwargs):
        super(RandomAgent, self).__init__()

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

        self.toggle_rate = 0.020

    def forward(self, obs):

        instances = obs.shape[0]
        action = 1.0 \
            * (torch.rand(instances,1,self.action_width, self.action_height)\
                <= self.toggle_rate)

        return action


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
        self.toggle_rate = 0.1



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

        action = 1.0 * (output <= self.toggle_rate)

        action = action.reshape(instances, 1, \
                self.action_width, self.action_height)

        return action


if __name__ == "__main__":
    
    #high life: 23/36
    #life: 23/3
    #mouse_maze: 12345/37
    #walled_cities: 2345/45678

    if(0):
        my_steps = 512

        for rules, name in zip([[[2,3],[3]], [[1,2,3,4,5],[3,7]]],\
                ["life", "mouse_maze"]):
            env_fn = AutomaticCellularEnvironment 
            env = RND2D(env_fn)

            env.env.survive = rules[0]
            env.env.birth = rules[1]
            env.env.instances = 1
            env.env.batch_size = 32


            action = torch.ones(env.env.instances,1,32,32)
            obs = env.reset()
            rewards = []
            steps = []
            for step in range(my_steps):

                obs, reward, done, info = env.step(action)

                rewards.append(reward.squeeze().detach().numpy())
                steps.append(step)

                #env.env.save_frame()

                fig = plt.figure(figsize=(6,12))
                plt.subplot(211)
                plt.plot(steps, rewards)
                plt.plot(steps[-1], rewards[-1], "o")
                plt.title("ones policy, RND reward in {} CA".format(name))
                plt.subplot(212)
                plt.imshow(obs[0,0,:,:].detach().numpy())
                plt.title("{} CA".format(name))
                plt.savefig("./frames/ones_rewards_{}_step{}".format(name, step))
                plt.close(fig)

    my_steps = 1024

    action = torch.ones(1,1,32,32)
    for rules, name in zip([[[2,3],[3]], [[1,2,3,4,5],[3,7]]],\
            ["life", "mouse_maze"]):
        agent = RandomAgent() 
        env_fn = AutomaticCellularEnvironment 
        env = RND2D(env_fn)

        env.env.survive = rules[0]
        env.env.birth = rules[1]
        env.env.instances = 1
        env.env.batch_size = 2


        obs = env.reset()
        rewards = []
        steps = []
        print("toggle rate: ", action.mean())
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
