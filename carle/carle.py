import numpy as np
import matplotlib.pyplot as plt
import os
import time 

import skimage
import skimage.io

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

class CARLE(nn.Module):

    def __init__(self, **kwargs):
        super(CARLE,self).__init__()

        self.inner_env = None
        self.width = kwargs["width"] if "width" in kwargs.keys() else 64
        self.height = kwargs["height"] if "height" in kwargs.keys() else 64

        self.action_width = 32 
        self.action_height = 32  

        self.alive_rate = kwargs["alive_rate"] if "alive_rate" in kwargs.keys()\
                else 0.0
        
        # instances define how many CA universes to run in parallel via vectorization 
        self.instances = kwargs["instances"] if "instances" in kwargs.keys()\
                else 1

        # keep track of universe development
        self.logging = kwargs["logging"] if "logging" in kwargs.keys() else False

        self.set_neighborhood()
        self.set_action_padding()

        # Conway's GoL rules
        self.birth = [3]
        self.survive = [2,3]


    def set_neighborhood(self):
        """
        Establish the neighborhood function as a convolutional layer
        Moore neighborhoods are used in Life-like CA
        """
        
        circular = True
        self.use_cuda = False

        moore_kernel = torch.tensor([[1.,1.,1.], [1.,0.,1.], [1.,1.,1.]],\
                requires_grad=False)

        if circular:
            my_mode = "circular"
        else:
            my_mode = "zeros"

        self.neighborhood = nn.Conv2d(1, 1, 3, padding=1,\
                padding_mode=my_mode, bias=False)


        if torch.cuda.is_available() and self.use_cuda:
            self.my_device = "cuda"
            self.neighborhood.to(self.my_device)
            self.to(self.my_device)

            # run on multiple gpus if possible
            #self.neighborhood = nn.DataParallel(self.neighborhood)
        else:
            self.my_device = "cpu"
            self.neighborhood.to(self.my_device)

        for param in self.neighborhood.parameters():
            param.requres_grad = False

        for param in self.neighborhood.named_parameters():
            param[1][0] = moore_kernel


        for param in self.neighborhood.parameters():
            param.requres_grad = False

    def set_action_padding(self):


        assymetry_width = (self.width - self.action_width) % 2
        assymetry_height = (self.height - self.action_height) % 2

        self.action_width -= (self.width % 2)
        self.action_height -= (self.width % 2)

        width_padding = (self.width - self.action_width) // 2 
        height_padding = (self.height - self.action_height) // 2

        self.action_padding = nn.ZeroPad2d(padding=\
                (width_padding, width_padding + assymetry_width,\
                height_padding, height_padding + assymetry_height))

    def reset(self):
        
        self.universe = 1.0 * \
                (torch.rand(self.instances, 1, self.width, self.height)\
                < self.alive_rate)

        self.universe = self.universe.to(self.my_device)
        observation = self.universe

        self.instance_id = str(int(time.time()))
        self.step_number = 0

        # used to determine when logging universe rle is necessary
        self.steps_since_action = 0
        self.log = []

        return observation

    def apply_action(self, action):

        if type(action) is not torch.Tensor:
            action = torch.Tensor(action)

        while len(action.shape) < 4:
            action = action.unsqueeze(0)

        action = action.to(self.my_device)

        # this may be better as an assertion line to avoid silent failures
        action = action[0, 0, :self.action_width, :self.action_height]

        action = self.action_padding(action)

        # toggle cells according to actions
        self.universe = 1.0 * torch.logical_xor(self.universe, action)

    def get_observation(self):

        return self.universe

    def step(self, action):
        
        if torch.sum(action):
            self.apply_action(action)

            if self.logging:
                self.log_universe()

        else:
            self.time_since_action += 1

        my_neighborhood = self.neighborhood(self.universe)

        universe_1 = torch.zeros_like(self.universe) 

        for b in self.birth:
            universe_1[((1-self.universe) * (my_neighborhood == b)) == 1] = 1

        for s in self.survive:
            universe_1[(self.universe * (my_neighborhood == s)) == 1] = 1

        
        self.universe = universe_1
        self.step_number += 1

        # This environment is open-ended free from exogenous reward,
        # giving no done signal and a reward of 0.0
        # episodic constraints and endogenous rewards have to be implemented
        # by wrappers or agents themselves.
        observation = self.get_observation()
        reward = torch.zeros(self.instances, 1).to(self.my_device)
        done = torch.zeros(self.instances, 1)
        info = [{}] * self.instances

        return observation, reward, done, info

    def render(self):

        os.system("clear")
        print("\n CA Universe")

        for ii in range(self.universe.shape[2]):
            print("")
            for jj in range(self.universe.shape[3]):
                if self.universe[0,0,ii,jj]:
                    print("o", end="")
                else:
                    print(" ", end="")

        time.sleep(0.125)
                #print(self.universe[0,0,ii,jj], end="\r")

    def get_rle(self, universe):

        "compute run-length encoding for given universe"


        #write header
        rle = "x = 0, y = 0, rule = B" 
        for bb in self.birth: rle += str(bb)
        rle += "/S"  
        for ss in self.survive: rle += str(ss)
        rle += ":T{}, {}\n".format(self.width, self.height)

        return rle

    def log_universe(self, universe_index=0):
        pass
        
    def save_log(self):
        pass

    def save_rle(self, rle):

        with open("./logs/universe{}_step{}.rle"\
                .format(self.instance_id, self.step_number), 'w') as f:

            f.write(rle)



    def save_frame(self):
        """
        save frames from the first instance of 'universe' 
        (at index [0,0,:,:] from self.universe tensor)
        """

            
        skimage.io.imsave("./frames/frame{}_step{}.png"\
                .format(self.instance_id, self.step_number), \
                np.uint8(255 * self.universe[0,0,:,:].detach().cpu().numpy()))




"""
Life-like CA rules
        if(rule=='conway'):
            #23/3
            # Conway's game of life rules (live cell)
            self.live_rules[0:2] = 0
            self.live_rules[2:4] = 1
            self.live_rules[4:] = 0

            # Conway's game of life rules (live cell)
            self.dead_rules[3] = 1
        elif(rule=='pseudo_life'):
            #238/357
            self.live_rules[2:4] = 1
            self.live_rules[8] = 1
            
            self.dead_rules[3] = 1
            self.dead_rules[5] = 1
            self.dead_rules[7] = 1

        elif(rule=='inverse_life'):
            #34678/0123478/2 
            #B012345678/S34678 << reverse of S/B format used elsewhere
            self.dead_rules[:] = 1

            self.live_rules[3:5] = 1
            self.live_rules[6:9] = 1
        elif(rule=='walled_cities'):
            #2345/45678
            self.live_rules[2:6] = 1

            self.dead_rules[4:9] = 1
        elif(rule=='maze'):
            #12345/3
            self.live_rules[1:6] = 1

            self.dead_rules[3] = 1
        elif(rule=='mouse_maze'):
            #12345/37
            self.live_rules[1:6] = 1

            self.dead_rules[3] = 1
            self.dead_rules[7] = 1
        elif(rule=='move'):
            #245/368
            self.live_rules[2] = 1
            self.live_rules[4] = 1
            self.live_rules[5] = 1
            
            self.dead_rules[3] = 1
            self.dead_rules[6] = 1
            self.dead_rules[8] = 1
        elif(rule=='replicator'):
            #1357/1357
            self.live_rules[1] = 1
            self.live_rules[3] = 1
            self.live_rules[5] = 1
            self.live_rules[7] = 1
            self.dead_rules[1] = 1
            self.dead_rules[3] = 1
            self.dead_rules[5] = 1
            self.dead_rules[7] = 1
        elif(rule=='2x2'):
            #125/36
            self.live_rules[1:3] = 1
            self.live_rules[5] = 1
            
            self.dead_rules[3] = 1
            self.dead_rules[6] = 1
        elif(rule=='34_life'):
            #34/34
            self.live_rules[3:5] = 1
            self.dead_rules[3:5] = 1
        elif(rule=='amoeba'):
            #1358/357
            self.live_rules[1] = 1
            self.live_rules[3] = 1
            self.live_rules[5] = 1
            self.live_rules[8] = 1
            
            self.dead_rules[3] = 1
            self.dead_rules[5] = 1
            self.dead_rules[7] = 1
        elif(rule=='diamoeba'):
            #5678/35678
            self.live_rules[5:9] = 1

            self.dead_rules[3] = 1
            self.dead_rules[5] = 1
            self.dead_rules[7] = 1
        elif(rule=='coral'):
            #45678/3
            self.live_rules[4:9] = 1
            
            self.dead_rules[3] = 1
        elif(rule=='coagulations'):
            #235678/378
            self.live_rules[2:4]
            self.live_rules[5:9] = 1
            
            self.dead_rules[3] = 1
            self.dead_rules[7] = 1
            self.dead_rules[8] = 1
            
        elif(rule=='gnarl'):
            #1/1
            self.live_rules[1] = 1
            
            self.dead_rules[1] = 1
        elif(rule=='assimilation'):
            #4567/345
            self.live_rules[4:8] = 1
            self.dead_rules[3:6] = 1
        elif(rule=='day_and_night'):
            #34678/3678
            self.live_rules[3:5] = 1
            self.live_rules[6:9] = 1

            self.dead_rules[3] = 1
            self.dead_rules[6:9] = 1
        elif(rule=='high_life'):
            #23/36
            # this rule has a replicator
            self.live_rules[2:4] = 1

            self.dead_rules[3] = 1
            self.dead_rules[6] = 1
        
"""


if __name__ == '__main__':

    env = CARLE()

    obs = env.reset()
    
    my_steps = 2048

    action = 1.0 * (torch.rand(env.instances,1,32,32) < 0.1)

    t0 = time.time()
    for step in range (my_steps):
        #env.render()
        _ = env.step(action)


    rle = env.get_rle(env.universe[0,0,:,:])

    env.save_rle(rle)
    env.save_frame()

    t1 = time.time()
    print("CA updates per second with {}x vectorization = {} and saving frames"\
            .format(env.instances, my_steps * env.instances/(t1-t0)))

    if(0):


        for instances in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            action = torch.ones(env.instances,1,32,32)
            env.instances = instances
            obs = env.reset()
            t2 = time.time()

            for step in range(my_steps):
                _ = env.step(action)
            

            t3 = time.time()
            print("CA updates per second with {}x vectorization = {}"\
                    .format(env.instances, my_steps * env.instances/(t3-t2)))




