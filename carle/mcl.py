"""
endogenous reward system for open-ended learning. 
    mcl stands for mesocorticolimbic system, aka the reward system in the human brain
"""
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from carle import CARLE

import matplotlib.pyplot as plt


class Motivator(nn.Module):

    def __init__(self, env, **kwargs):
        super(Motivator, self).__init__()
        
        self.env = env

    def reset(self):

        obs = self.env.reset()
        
        return obs

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        return obs, reward, done, info

class RND2D(Motivator):
    """
    An implentation of random network distillation (Burda et al. 2018) 
    for 4d tensors representing 2D cellular automata universes
    """
    def __init__(self, env, **kwargs):
        super(RND2D, self).__init__(env, **kwargs)
        
        self.my_name = "RND2D"

        self.learning_rate = 1e-3

        self.reward_scale = 1.0
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

        reward += self.reward_scale * rnd_bonus

        return obs, reward, done, info

    def reset(self):

        self.initialize_predictor()
        self.initialize_random_network()

        obs = self.env.reset()

        self.to(self.env.my_device)

        self.updates = 0

        return obs


class AE2D(RND2D):

    def __init__(self, env, **kwargs):
        super(AE2D, self).__init__(env_fn, **kwargs)
        self.learning_rate = 1e-3

        self.my_name = "AE2D"


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

class SpeedDetector(Motivator):

    def __init__(self, env, **kwargs):
        super(SpeedDetector, self).__init__(env, **kwargs)

        self.reward_scale = 1.0

        self.com = np.array([(self.env.height - 1) / 2.0,\
                            (self.env.width - 1) / 2.0])

        self.speed_modulator = 32.0

        self.mass_weight_h = torch.arange(self.env.height).reshape(1, -1)
        self.mass_weight_w = torch.arange(self.env.width).reshape(-1, 1)

        self.growing_steps = 0
        self.smooth_velocity = 0.0
        self.speed = 0.0

        self.velocity = np.array([0.0, 0.0])

        self.live_cells = 0

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        live_cells = torch.sum(self.env.universe)
        center_of_mass_h = torch.sum(self.env.universe * self.mass_weight_h)\
                / live_cells
        center_of_mass_w = torch.sum(self.env.universe * self.mass_weight_w)\
                / live_cells

        com = np.array([center_of_mass_h, center_of_mass_w])

        if not(torch.sum(action)):

            self.growing_steps += 1

            alpha = 1. / self.speed_modulator

            velocity = self.com - com

            # exponential average of velocity
            self.smooth_velocity = (1. - alpha) * self.smooth_velocity + alpha * velocity 

            self.speed = np.sqrt( np.sum(self.smooth_velocity)**2 )
            self.velocity = velocity

            reward += self.reward_scale * self.speed
            

        else:

            self.growing_steps = 0
            self.speed = 0.0 

        self.live_cells = live_cells

        return obs, reward, done, info


class PufferDetector(Motivator):

    """
    This wrapper detects unbounded growth patterns in the absence of agent actions.
    Due to the simplicity of the detection logic, it will also detect glider/spaceship guns
    and naturally ocurring growth (i.e. growth due to growing rulesets like B3/45678.

    In cases where growth is naturally ocurring, this wrapper would  
    """

    def __init__(self, env, **kwargs):
        super(PufferDetector, self).__init__(env, **kwargs) 

        self.my_name = "PufferDetector"

        self.live_cells = 0
        self.reward_scale = 1.0
        
        # growth threshold is also used to calculate the exponential average num live cells
        self.growth_threshold = 16

        self.growing_steps = 0
    
    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        live_cells = torch.sum(self.env.universe).cpu().numpy() 

        
        if not(torch.sum(action)):

            if live_cells > self.live_cells:
                self.growing_steps += 1
            else: 
                self.growing_steps = 0


            if self.growing_steps > self.growth_threshold: 
                reward += self.reward_scale
        else:

            self.growing_steps = 0
        
        alpha = 1. / self.growth_threshold
        self.live_cells = (1 - alpha) * self.live_cells + alpha * live_cells

        return obs, reward, done, info


if __name__ == "__main__":


    env_fn = CARLE() 
    env = SpeedDetector(env_fn)

    #rules for Life
    env.env.birth = [3]
    env.env.survive = [2,3]


    # no growth reward

    obs = env.reset()
    sum_reward = 0.0
    for step in range(64):

        action = torch.ones(env.env.instances,\
                1, env.env.action_height, \
                env.env.action_height)

        obs, r, d, i = env.step(action)
        
        sum_reward += r

    print("sum of rewards with toggles ", sum_reward)

    # growth reward (no toggles)
    obs = env.reset()
    sum_reward = 0.0
    env.step(action)
    for step in range(64):

        action = torch.zeros(env.env.instances, \
                1, env.env.action_height, \
                env.env.action_height)

        obs, reward, d, i = env.step(action)

        
        sum_reward += reward

    print("sum of rewards without toggles ", sum_reward)


    # growth reward (no toggles)
    obs = env.reset()
    sum_reward = 0.0

    action = torch.zeros(env.env.instances, \
            1, env.env.action_height, \
            env.env.action_height)
    action[:,:,14, 16] = 1.0
    action[:,:,15, 16:18] = 1.0
    action[:,:,16, 15:18:2] = 1.0

    env.step(action)

    for step in range(64):

        action = torch.zeros(env.env.instances, \
                1, env.env.action_height, \
                env.env.action_height)

        obs, reward, d, i = env.step(action)

        
        sum_reward += reward

    print("sum of rewards with glider ", sum_reward)
    

