"""
endogenous reward system for open-ended learning. 
    mcl stands for mesocorticolimbic system, aka the reward system in the human brain
"""
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from carle import AutomaticCellularEnvironment

import matplotlib.pyplot as plt


class Motivator(nn.Module):

    def __init__(self, env_fn, **kwargs):
        super(Motivator, self).__init__()
        
        self.env = env_fn(**kwargs)

    def reset(self):

        obs = self.env.reset
        
        return obs

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        return obs, reward, done, info

class RND2D(Motivator):
    """
    An implentation of random network distillation (Burda et al. 2018) 
    for 4d tensors representing 2D cellular automata universes
    """
    def __init__(self, env_fn, **kwargs):
        super(RND2D, self).__init__(env_fn, **kwargs)
        
        self.learning_rate = 1e-3
        self.curiosity_scale = 1.0
        self.rnd_dim = 16

        self.initialize_predictor()
        self.initialize_random_network()

        self.buffer_length = 0
        self.batch_size = 8

        self.updates = 0

    def initialize_predictor(self):

        dense_nodes = (self.env.width // 8) * (self.env.height // 8)

        self.predictor = nn.Sequential(\
                nn.Conv2d(1, 8, 3, padding=1, stride=1),\
                nn.ReLU(),\
                nn.MaxPool2d(2,2,padding=0),\
                nn.MaxPool2d(2,2,padding=0),\
                nn.Conv2d(8, 1, 3, padding=1, stride=1),\
                nn.ReLU(),\
                nn.MaxPool2d(2,2,padding=0),\
                nn.Flatten(),\
                nn.Linear(dense_nodes, self.rnd_dim),\
                nn.Tanh()\
                )

        self.optimizer = torch.optim.Adam(self.predictor.parameters(),\
                lr=self.learning_rate)

    def initialize_random_network(self):

        dense_nodes = (self.env.width // 8) * (self.env.height // 8)

        self.random_network = nn.Sequential(\
                nn.Conv2d(1, 4, 3, padding=1, stride=1),\
                nn.ReLU(),\
                nn.MaxPool2d(2,2,padding=0),\
                nn.MaxPool2d(2,2,padding=0),\
                nn.Conv2d(4, 1, 3, padding=1, stride=1),\
                nn.ReLU(),\
                nn.MaxPool2d(2,2,padding=0),\
                nn.Flatten(),\
                nn.Linear(dense_nodes, self.rnd_dim),\
                nn.Tanh()\
                )

        for param in self.random_network.parameters():
            param.requires_grad = False
            

    def random_forward(self, obs):

        return self.random_network(obs)

    def forward(self, obs):
        """
        corresponds to forward pass of the predictor network
        """

        x = self.predictor(obs)

        return x


    def update_predictor(self, loss):

        loss.backward()

        self.optimizer.step()

        self.updates += 1

    def get_bonus(self, obs):

        target = self.random_forward(obs)
        prediction = self.forward(obs)

        loss = torch.mean(torch.abs((target-prediction)**2),\
                dim=[1])

        # loss is a tensor used for vectorized rnd reward bonuses
        return loss

    def get_bonus_update(self, obs):

        self.predictor.zero_grad()

        # loss is a tensor used for vectorized rnd reward bonuses
        loss = self.get_bonus(obs)

        # update loss is a scalar used for backprop
        update_loss = torch.mean(loss)

        self.update_predictor(update_loss)

        return loss

    def get_bonus_accumulate(self, obs):

        if self.buffer_length == 0:
            self.predictor.zero_grad()
            self.accumulate_loss = 0.0

        loss = self.get_bonus(obs)

        self.accumulate_loss += torch.mean(loss)

        self.buffer_length += 1

        if self.buffer_length >= self.batch_size:
            self.accumulate_loss = self.accumulate_loss / self.batch_size

            self.update_predictor(self.accumulate_loss)
            self.buffer_length = 0

        return loss


    def get_bonus_only(self, obs):

        with torch.no_grad():

            loss = self.get_bonus(obs)

        return loss


    def step(self, action):

        action = action.to(self.env.my_device)
        obs, reward, done, info = self.env.step(action)

        rnd_bonus = self.get_bonus_accumulate(obs).unsqueeze(1)

        reward += self.curiosity_scale * rnd_bonus

        return obs, reward, done, info

    def reset(self):

        self.initialize_predictor()
        self.initialize_random_network()

        obs = self.env.reset()

        self.to(self.env.my_device)

        self.updates = 0

        return obs


class AE2D(RND2D):

    def __init__(self, env_fn, **kwargs):
        super(AE2D, self).__init__(env_fn, **kwargs)
        self.learning_rate = 1e-3

    def initialize_random_network(self):
        pass

    def forward(self, obs):

        prediction = self.predictor(obs)

        prediction = prediction.reshape(\
                self.env.instances, 1, self.env.height, self.env.width)

        return prediction

    def initialize_predictor(self):

        dense_in = (self.env.width // 8) * (self.env.height // 8)
        dense_out = (self.env.width) * (self.env.height)

        if(0):
            self.predictor = nn.Sequential(\
                    nn.Conv2d(1, 8, 3, padding=1, stride=1),\
                    nn.ReLU(),\
                    nn.MaxPool2d(2,2,padding=0),\
                    nn.Conv2d(8, 2, 3, padding=1, stride=1),\
                    nn.ReLU(),\
                    nn.MaxPool2d(2,2,padding=0),\
                    nn.ConvTranspose2d(2, 1, 4, padding=1, stride=2),\
                    nn.ReLU(),\
                    nn.ConvTranspose2d(1, 1, 4, padding=1, stride=2),\
                    nn.Sigmoid()\
                    )
        else:
            self.predictor = nn.Sequential(\
                    nn.Flatten(),\
                    nn.Linear(dense_out, 1024, bias=False),\
                    nn.ReLU(),\
                    nn.Linear(1024, 256, bias=False),\
                    nn.ReLU(),\
                    nn.Linear(256, dense_out, bias=False),\
                    nn.Sigmoid()\
                    )


        self.optimizer = torch.optim.Adam(self.predictor.parameters(),\
                lr=self.learning_rate)

    def update_predictor(self, loss):

        loss.backward()

        self.optimizer.step()

    def get_bonus(self, obs):

        prediction = self.forward(obs)

        loss = torch.mean(torch.abs((obs-prediction)**2),\
                dim=[1,2,3])

        # loss is a tensor used for vectorized rnd reward bonuses
        return loss

    def get_bonus_update(self, obs):

        self.predictor.zero_grad()

        # loss is a tensor used for vectorized rnd reward bonuses
        loss = self.get_bonus(obs)

        # update loss is a scalar used for backprop
        update_loss = torch.mean(loss)

        self.update_predictor(update_loss)

        return loss

    def get_bonus_accumulate(self, obs):

        if self.buffer_length == 0:
            self.predictor.zero_grad()
            self.accumulate_loss = 0.0

        loss = self.get_bonus(obs)

        self.accumulate_loss += torch.mean(loss)

        self.buffer_length += 1

        if self.buffer_length >= self.batch_size:
            self.accumulate_loss = self.accumulate_loss / self.batch_size

            self.update_predictor(self.accumulate_loss)

            self.buffer_length = 0

        return loss


    def get_bonus_only(self, obs):

        with torch.no_grad():

            loss = self.get_bonus(obs)

        return loss


if __name__ == "__main__":



    env_fn = AutomaticCellularEnvironment 
    env = RND2D(env_fn)

    number_steps = 500

    rewards = {}

    plt.figure()
    for batch_size in [1, 8, 16]:

        for wrapper, name in zip([AE2D, RND2D], ["AE", "RND"]):
            action = torch.zeros(env.env.instances, 1, \
                   env.env.action_width, env.env.action_height)
    
            action[0,0,4:6,16] = 1.0
            action[0,0,6,15] = 1.0
            action[0,0,6,17] = 1.0
            action[0,0,7:11,16] = 1.0
            action[0,0,11,15] = 1.0
            action[0,0,11,17] = 1.0
            action[0,0,12:14,16] = 1.0

            env = wrapper(env_fn)
            env.batch_size = batch_size
            name += str(batch_size)
            rewards[name] = []

            cumulative_reward = 0.0
            obs = env.reset()

            for my_step in range(number_steps):

                obs, reward, done, info = env.step(action)
                action *= 0

                cumulative_reward += reward
                rewards[name].append(reward[0,0].detach().cpu().numpy())
            
            print("cumulative reward = {}".format(cumulative_reward))

            plt_index = 1 + 1 * ("rnd" in name.lower())
            plt.subplot(1,2,plt_index)
            plt.plot(rewards[name], label=name)
            plt.legend()
            plt.title("{} exploration bonus ".format(name))
        
            action = torch.ones(env.env.instances,1,32,32)


            if(1):
                print("throughput with {} bonus wrapper:".format(name))

                my_steps = 4096
                obs = env.reset()

                env.batch_size = batch_size
                print("batch size = {}".format(env.batch_size))
                for instances in [1, 16, 256]:
                    env.env.instances = instances
                    action = torch.ones(env.env.instances,1,32,32)
                    obs = env.reset()
                    t2 = time.time()

                    for step in range(my_steps):
                        _ = env.step(action)
                    

                    t3 = time.time()
                    print("CA updates per second with {}x vectorization = {}"\
                            .format(env.env.instances, my_steps * env.env.instances/(t3-t2)))

    plt.legend()
    plt.show()
