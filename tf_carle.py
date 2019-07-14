from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment

from tf_agents.trajectories import time_step as ts

from tf_agents.environments import utils
from tf_agents.specs import array_spec

tf.compat.v1.enable_v2_behavior()

import matplotlib.pyplot as plt

class CAEnv(py_environment.PyEnvironment):
    def __init__(self):
            
        dim_x=64
        dim_y=64
        dim_distillate=16
             
        # universe dimensions    
        self.dim_x = dim_x
        self.dim_y = dim_y
        
        # maximum episode length
        self.max_steps = 2048

        # we don't need to specify the random network within the environment
        # move rnd entirely outside environment
        self.dim_distillate = dim_distillate
        
        #reward weighting - also don't need this
        self.w_rnd = 1.0
        self.w_der = 1.0 - self.w_rnd
        
        self.reset()
        self.distiller = None
         
       
        # actions consist of an array of toggle instructions with the same size
        # as the observation space,  
        self._action_spec = array_spec.BoundedArraySpec(\
                shape=(1, self.dim_x, self.dim_y),\
                dtype=np.int32, minimum = 0, maximum = 1, name='action')

        self._observation_spec = array_spec.BoundedArraySpec(\
                shape=(1, self.dim_x, self.dim_y), minimum = 0, maximum=1,\
                dtype = np.int32, name='observation')

        self._episode_ended = False

        # begin with gosper glider generator as initial state
        init_state = (np.load('./data/init_state.npy')).astype(np.int32) 
        #init_state /= 255
        self._state = init_state
        
    
        self.episode_step = 0
        self.plane_memory = np.zeros_like(self._state)
        #determine rules
        self.live_rules = np.zeros((9,)) 
        self.dead_rules = np.zeros((9,)) 
        rule = 'conway'        
        if(rule=='conway'):
            #23/3
            # Conway's game of life rules (live cell)
            self.live_rules[0:2] = 0
            self.live_rules[2:4] = 1
            self.live_rules[4:] = 0
            # Conway's game of life rules (dead cell)
            self.dead_rules[3] = 1
 
    def action_spec(self):   
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):

        # begin with gosper glider generator as initial state
        init_state = (np.load('./data/init_state.npy')).astype(np.int32) 
        self._state = init_state
        
        self._episode_ended = False
        self.episode_step = 0

        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):
        """
        update ca grid and return observation/reward
        """
        sum_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
        
        print(action.shape)
        plane = (self._state | action) - (self._state & action)
        
        new_plane = np.copy(self._state)

        print('update ca world') 
        for xx in range(self.dim_x):
            for yy in range(self.dim_y):
                temp_sum = 0
                if xx == 0: 
                    ii_prev = self.dim_x-1
                    ii_next = xx + 1
                elif xx == (self.dim_x-1):
                    ii_prev = xx - 1
                    ii_next = 0
                else:
                    ii_prev = xx-1
                    ii_next = xx+1
                    
                if yy == 0:
                    jj_prev = self.dim_y-1
                    jj_next = yy + 1
                elif yy == (self.dim_y-1):
                    jj_prev = yy-1
                    jj_next = 0
                else:
                    jj_prev = yy - 1
                    jj_next = yy + 1
                
                ii, jj = xx, yy
                
                # get the row above
                temp_sum += np.sum(plane[ii_prev,jj_prev])
                temp_sum += np.sum(plane[ii_prev,jj])
                temp_sum += np.sum(plane[ii_prev,jj_next])
                # get the row below
                temp_sum += np.sum(plane[ii_next,jj_prev])
                temp_sum += np.sum(plane[ii_next,jj])
                temp_sum += np.sum(plane[ii_next,jj_next])
                # get the current row
                temp_sum += np.sum(plane[ii,jj_prev] + plane[ii,jj_next])


                if (plane[ii,jj]):
                    new_plane[xx,yy] = self.live_rules[int(temp_sum)]
                else:
                    new_plane[xx,yy] = self.dead_rules[int(temp_sum)]
        
        self._state= new_plane

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        if self.episode_step > self.max_steps:

            self._episode_ended = True
        
        self.episode_step += 1

        reward = - np.mean(action)

        if self._episode_ended:
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(
              np.array([self._state], dtype=np.int32), reward=reward, discount=1.0)

if __name__ == '__main__':
    env = CAEnv()
    
    #   utils.validate_py_environment(env, episodes=1)
    for my_step in range(10):
        print('do stuff')   
        action = np.zeros((1, env.dim_x, env.dim_y))
    
        action[0,   np.random.randint(64), np.random.randint(64)] = 1

        step = env._step(np.random.randint(2, size=(64,64)))
        adj_reward = step.reward + 10
        print('reward: {}/{}'.format(step.reward, adj_reward))
