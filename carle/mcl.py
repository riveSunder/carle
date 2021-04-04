"""
endogenous reward system for open-ended learning. 
    mcl stands for mesocorticolimbic system, aka the reward system in the human brain
    
    These reward system classes act as wrappers around CARLE, and are invoked as

    `
    env = MotivatorClass(env)
    `

    Multiple wrappers can be applied in series, and the original CARLE environment can be 
    accessed as `env.inner_env`
"""
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import carle
from carle.env import CARLE

import matplotlib.pyplot as plt


class Motivator(nn.Module):

    def __init__(self, env, **kwargs):
        super(Motivator, self).__init__()
        
        if env.inner_env == None:
            self.inner_env = env
        else:
            self.inner_env = env.inner_env

        self.env = env

    def reset(self):

        obs = self.env.reset()
        
        return obs

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        return obs, reward, done, info

    def set_no_grad(self):

        pass

    def set_grad(self):

        pass

class RND2D(Motivator):
    """
    An implentation of random network distillation (Burda et al. 2018) 
    for 4d tensors representing 2D cellular automata universes
    """
    def __init__(self, env, **kwargs):
        super(RND2D, self).__init__(env, **kwargs)
        
        self.my_name = "RND2D"

        self.learning_rate = 6e-2

        self.reward_scale = 1.0
        self.rnd_dim = 16

        self.initialize_predictor()
        self.initialize_random_network()

        self.buffer_length = 0
        self.batch_size = 64

        self.updates = 0


    def initialize_predictor(self):

        dense_nodes = (self.inner_env.width // 8) * (self.inner_env.height // 8)

        self.predictor = nn.Sequential(\
                nn.Conv2d(1, 4, 3, padding=1, stride=1),\
                nn.Dropout(p=0.1),\
                nn.ReLU(),\
                nn.MaxPool2d(2,2,padding=0),\
                nn.MaxPool2d(2,2,padding=0),\
                nn.Conv2d(4, 1, 3, padding=1, stride=1),\
                nn.Dropout(p=0.1),\
                nn.ReLU(),\
                nn.MaxPool2d(2,2,padding=0),\
                nn.Dropout(p=0.1),\
                nn.Flatten(),\
                nn.Linear(dense_nodes, self.rnd_dim),\
                nn.Tanh()\
                )

        self.optimizer = torch.optim.Adam(self.predictor.parameters(),\
                lr=self.learning_rate)

    def initialize_random_network(self):

        dense_nodes = (self.inner_env.width // 8) * (self.inner_env.height // 8)

        self.random_network = nn.Sequential(\
                nn.Conv2d(1, 2, 3, padding=1, stride=1),\
                nn.ReLU(),\
                nn.MaxPool2d(2,2,padding=0),\
                nn.MaxPool2d(2,2,padding=0),\
                nn.Conv2d(2, 1, 3, padding=1, stride=1),\
                nn.ReLU(),\
                nn.MaxPool2d(2,2,padding=0),\
                nn.Flatten(),\
                nn.Linear(dense_nodes, self.rnd_dim),\
                nn.Tanh()\
                )

        self.set_grad()

    def set_grad(self):


        for param in self.predictor.parameters():
            param.requires_grad = True

        for param in self.random_network.parameters():
            param.requires_grad = False
            

    def set_no_grad(self):


        for param in self.predictor.parameters():
            param.requires_grad = False

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

        self.optimizer.zero_grad()

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

        action = action.to(self.inner_env.my_device)
        obs, reward, done, info = self.env.step(action)

        rnd_bonus = self.get_bonus_accumulate(obs).unsqueeze(1)

        reward += self.reward_scale * rnd_bonus

        return obs, reward, done, info

    def reset(self):

        self.initialize_predictor()
        self.initialize_random_network()

        obs = self.env.reset()

        self.to(self.inner_env.my_device)

        self.updates = 0

        return obs


class AE2D(RND2D):

    def __init__(self, env, **kwargs):
        super(AE2D, self).__init__(env, **kwargs)
        self.learning_rate = 9e-2

        self.my_name = "AE2D"


    def initialize_random_network(self):
        pass

    def forward(self, obs):

        prediction = self.predictor(obs)

        prediction = prediction.reshape(\
                self.inner_env.instances, 1, self.inner_env.height, self.inner_env.width)

        return prediction

    def initialize_predictor(self):

        dense_in = (self.inner_env.width // 8) * (self.inner_env.height // 8)
        dense_out = (self.inner_env.width) * (self.inner_env.height)

        if(1):
            self.predictor = nn.Sequential(\
                    nn.Conv2d(1, 4, 3, padding=1, stride=1),\
                    nn.Dropout(p=0.1),\
                    nn.ReLU(),\
                    nn.MaxPool2d(2,2,padding=0),\
                    nn.Conv2d(4, 2, 3, padding=1, stride=1),\
                    nn.Dropout(p=0.1),\
                    nn.ReLU(),\
                    nn.MaxPool2d(2,2,padding=0),\
                    nn.ConvTranspose2d(2, 1, 4, padding=1, stride=2),\
                    nn.Dropout(p=0.1),\
                    nn.ReLU(),\
                    nn.ConvTranspose2d(1, 1, 4, padding=1, stride=2),\
                    nn.Dropout(p=0.1),\
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

    def set_grad(self):

        

        for param in self.predictor.parameters():
            param.requires_grad = True


    def set_no_grad(self):



        for param in self.predictor.parameters():
            param.requires_grad = False

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

        self.com = None 

        self.speed_modulator = 32.0

        self.mass_weight_h = torch.arange(self.inner_env.height).reshape(1, -1)
        self.mass_weight_w = torch.arange(self.inner_env.width).reshape(-1, 1)

        self.growing_steps = 0
        self.smooth_velocity = None
        self.speed = None

        self.velocity = np.array([0.0, 0.0])

        self.live_cells = None

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        live_cells = torch.sum(self.inner_env.universe)
        center_of_mass_h = torch.sum(self.inner_env.universe * self.mass_weight_h)\
                / live_cells
        center_of_mass_w = torch.sum(self.inner_env.universe * self.mass_weight_w)\
                / live_cells

        com = np.array([center_of_mass_h, center_of_mass_w])

        if not(torch.sum(action)):

            alpha = 1. / self.speed_modulator

            if self.com is None:
                self.com = com
            else:
                velocity = self.com - com

                # exponential average of velocity
                if self.smooth_velocity is None:
                    self.smooth_velocity = velocity
                else:
                    self.smooth_velocity = (1. - alpha) * self.smooth_velocity + alpha * velocity 

                speed = np.sqrt(np.sum((velocity)**2))
                self.speed = np.sqrt( np.sum((self.smooth_velocity)**2) )
                self.velocity = velocity

                if self.com[0]  == com[0] and self.com[1] == com[1]:
                    self.speed = 0.0
                self.com = com

                if self.speed > 1 / (self.speed_modulator):
                    self.growing_steps += 1
                else:
                    self.growing_steps = 0

                if self.growing_steps > (self.speed_modulator / 2):

                    # intended to give a reward for any pattern traveling at c/(speed_modulator) or faster
                    # for at least speed_modulator/2 steps
                    # and weighted by the size of the spaceship

                    reward += self.reward_scale * live_cells
            
                

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

        self.cells = []
        self.live_cells = 0.0
        self.reward_scale = 1.0
        
        # growth threshold is also used to calculate the exponential average num live cells
        self.growth_threshold = 512

        self.growing_steps = 0
    
    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        self.live_cells = torch.sum(self.inner_env.universe).cpu().numpy() 

        
        if not(torch.sum(action)):

            self.cells.append(self.live_cells)

            if len(self.cells) > self.growth_threshold:
                slope = self.cells[-1] - self.cells[0]

                self.cells.pop(0)
                if slope > 0.01:
                    reward += 1


        else:
            self.growing_steps = 0
            self.cells = []
        

        
        return obs, reward, done, info

def get_symmetric_action(probability=0.125, vertical_symmetry=False):
    
    action = torch.zeros(0,0,64,64)
    midpoint = action.shape[3] // 2
    
    for ii in range(action.shape[2]):
        for jj in range(1, midpoint):
            
            if torch.rand(1) <= probability:
                if jj > 1:
                    
                    action[:,:,ii,midpoint+jj] = 1.0
                    action[:,:,ii,midpoint-jj] = 1.0
            
    
    return action

def get_glider():
    action = torch.zeros(1,1,64,64)
    action[:,:,32,32] = 1.0
    action[:,:,33,32:34] = 1.0
    action[:,:, 34, 31] = 1.0
    action[:,:, 34, 33] = 1.0
    
    return action

def get_morley_puffer():
    action = torch.zeros(1,1,64,64)
    action[:,:,31:33,32] = 1.0
    action[:,:,30,33] = 1.0
    action[:,:,33,33] = 1.0
    
    action[:,:,29:35,34] = 1.0
    
    action[:,:,30,35:37] = 1.0
    action[:,:,33,35:37] = 1.0
    action[:,:,31:33,37] = 1.0
    
    return action

if __name__ == "__main__":


    env = CARLE() 
    env = PufferDetector(env)
    #env = SpeedDetector(env)

    #rules for Life
    env.inner_env.birth = [3,6,8]
    ss = [2,4,5]

    action_fn = get_morley_puffer
    print("survival rules are ", ss)
    print(action_fn)
    env.inner_env.survive = ss

    for action_fn in [get_glider, get_morley_puffer, get_symmetric_action]:

        obs = env.reset()
        sum_reward = 0.0
        rewards = []
        cells = []
        
        action = action_fn()

        for step in range(3400):

            obs, reward, d, i = env.step(action)
            rewards.append(reward)
            
            cells.append(env.live_cells)

            action *= 0.0

            sum_reward += reward

        print("reward sum ", sum_reward)
        
        plt.figure()
        plt.plot(rewards)
        plt.plot(cells[1:]/ np.max(cells[1:]))
        
    env.reward_scale = 0.0
    env = SpeedDetector(env)
    env.inner_env.birth = [3]
    ss = [2,3]

    action_fn = get_glider
    print("survival rules are ", ss)
    print(action_fn)
    env.inner_env.survive = ss


    for action_fn in [get_glider, get_morley_puffer, get_symmetric_action]:
        obs = env.reset()
        sum_reward = 0.0
        rewards = []
        cells = []
        
        action = action_fn()

        for step in range(3160):

            obs, reward, d, i = env.step(action)
            rewards.append(reward)
            
            cells.append(env.speed)

            action *= 0.0

            sum_reward += reward

        print("reward sum ", sum_reward)
        
        plt.figure()
        plt.plot(rewards)
        plt.plot(cells[1:]/ np.max(cells[1:]))
    plt.show()


