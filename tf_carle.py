from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import network, encoding_network
from tf_agents.utils import nest_utils
from tf_agents.utils import common as common_utils
from tf_agents.environments import utils
from tf_agents.networks import utils as net_utils
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

tf.compat.v1.enable_v2_behavior()

# PPO stuff
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.agents.ppo.ppo_agent import PPOAgent
    
#FLAGS = absl.FLAGS

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Reshape, \
                                    AveragePooling1D, AveragePooling2D, Activation, \
                                    Dropout, Dense, Flatten, Lambda 

import matplotlib.pyplot as plt

from models import get_alexnet

class ActorNetwork(network.Network):

  def __init__(self,
               observation_spec,
               action_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               fc_layer_params=(75, 40),
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               enable_last_layer_zero_initializer=False,
               name='ActorNetwork'):
    super(ActorNetwork, self).__init__(
        input_tensor_spec=observation_spec, state_spec=(), name=name)
    """
    Actor network based on https://github.com/tensorflow/agents/blob/master/tf_agents/colabs/8_networks_tutorial.ipynb
    commit 51bdd32abd8003d9de8618cff599cf2a44a38cd6
    """

    self._action_spec = action_spec
    flat_action_spec = tf.nest.flatten(action_spec)

    kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1. / 3., mode='fan_in', distribution='uniform')

    self._encoder = encoding_network.EncodingNetwork(
        observation_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        dropout_layer_params=dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=False)

    initializer = tf.keras.initializers.RandomUniform(
        minval=-0.003, maxval=0.003)

    self._action_projection_layer = tf.keras.layers.Dense(
        flat_action_spec[0].shape.num_elements(),
        activation=tf.keras.activations.sigmoid,
        kernel_initializer=initializer,
        name='action')

  def call(self, observations, step_type=(), network_state=()):
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
    # We use batch_squash here in case the observations have a time sequence
    # compoment.
    batch_squash = net_utils.BatchSquash(outer_rank)
    observations = tf.nest.map_structure(batch_squash.flatten, observations)

    state, network_state = self._encoder(
        observations, step_type=step_type, network_state=network_state)
    actions = self._action_projection_layer(state)
    actions = common_utils.scale_to_spec(actions, self._action_spec)
    actions = batch_squash.unflatten(actions)

    return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state


class CAEnv(py_environment.PyEnvironment):
    def __init__(self):
            
        dim_x=64
        dim_y=64
        dim_distillate=16
             
        # universe dimensions    
        self.dim_x = dim_x
        self.dim_y = dim_y
        
        # maximum episode length
        self.max_steps = 256 

        # we don't need to specify the random network within the environment
        # move rnd entirely outside environment
        self.dim_distillate = dim_distillate
        
        #reward weighting - also don't need this
        self.w_rnd = 1.0
        self.w_der = 1.0 - self.w_rnd
        
        self.reset()
        self.rn = self.get_rn(self.dim_x, self.dim_y, self.dim_distillate)
        self.predictor = self.get_predictor(self.dim_x, self.dim_y, self.dim_distillate)
        
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
        
        #prediction = action[:,-16:]
        #action = action[:,:-16]
        # this is probably inefficient, but we need floats for the rnd prediction

        action = action.reshape(self.dim_x, self.dim_y)

        #action = np.array(action, dtype=np.int32)
        plane = (self._state | action) - (self._state & action)
        
        new_plane = np.copy(self._state)

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
        
        self._state = new_plane

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        if self.episode_step > self.max_steps:

            self._episode_ended = True
        
        # -- do rnd stuff
        rn_out = self.rn.predict(self._state[tf.newaxis,:,:])
        new_plane = self._state[tf.newaxis,:,:,tf.newaxis]
        try: 
            
            self.new_planes = np.apped(self.new_planes, new_plane, axis=0) 
            #old_planes = np.append(old_planes, old_plane, axis=0)
            self.distillates = np.append(self.distillates, rn_out, axis=0)
            if distillates.shape[0] > 3000:
                distillates = distillates[:3000,...]

                old_planes = old_planes[:3000,...]
                new_planes = new_planes[:3000,...]
        except:
            self.new_planes = new_plane
            #old_planes = old_plane
            self.distillates = rn_out

        prediction = self.predictor.predict(self._state[tf.newaxis,:,:,tf.newaxis])
        #start_loss = predictor.evaluate(new_planes, distillates, verbose=0)
        self.predictor.fit(self.new_planes, self.distillates, batch_size=64, epochs=3, verbose=0)
        #end_loss = predictor.evaluate(new_planes, distillates, verbose=0)

        # -- end do rnd stuff

        reward = - np.mean(action)
        
        rn_reward = np.sum(np.abs(prediction - rn_out)**2)
        reward += rn_reward


        self.episode_step += 1

        if self._episode_ended:
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(
              np.array([self._state], dtype=np.int32), reward=reward, discount=1.0)

    def get_rn(self, dim_x=64, dim_y=64, dim_out=16):
        """
        define random network (fixed weight mlp)
        """
        random_seed = 29

        model = Sequential()
        #model.add(Lambda(lambda x: x.reshape(1, dim_x, dim_y, 1)))
        model.add(Lambda(lambda x: tf.cast(x,tf.float32)))
        model.add(Flatten())
        
        model.add(Dense(512,\
                kernel_initializer=tf.keras.initializers.RandomNormal(\
                mean=0.0, stddev=0.5, seed=random_seed), activation='relu'))
        model.add(Dense(dim_out,\
                kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed)))

        model.add(Activation(tf.nn.sigmoid))
        model.add(Reshape([16]))
        model.trainable = False

        return model

    def get_predictor(self, dim_x=64, dim_y=64, dim_pred=16):
        """
        define model for predicting random network output
        """
        
        model = get_alexnet(dropout_rate=0.35)
        model.compile(loss='mse', optimizer=Adam(lr=1e-5))
        return model

if __name__ == '__main__':
    env = CAEnv()
    
    tf_env = tf_py_environment.TFPyEnvironment(env)

    train_env = tf_py_environment.TFPyEnvironment(env)
    eval_env = tf_py_environment.TFPyEnvironment(env)
    #   utils.validate_py_environment(env, episodes=1)
    #predictor = get_alexnet(dropout_rate=0.35) #alex.get_model()
    #predictor.compile(loss='mse', optimizer=Adam(lr=1e-5), metrics=['acc'])
    
    fc_layer_params = (128,64)
    my_net = ActorNetwork(\
            tf_env.observation_spec(),\
            tf_env.action_spec(),\
            fc_layer_params=fc_layer_params\
            )

    my_net = ActorDistributionNetwork(\
            tf_env.observation_spec(),\
            tf_env.action_spec(),\
            fc_layer_params=fc_layer_params\
            )

    
    time_step = tf_env.reset()
   
    lr = 1e-4
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

    train_step_counter = tf.compat.v2.Variable(0)

    ppo_agent = PPOAgent(\
            tf_env.time_step_spec(),\
            tf_env.action_spec(),\
            actor_net=my_net,\
            optimizer=optimizer,\
            train_step_counter=train_step_counter\
            )

    ppo_agent.initialize()

    eval_policy = ppo_agent.policy
    collect_policy = ppo_agent.collect_policy
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    def compute_avg_return(environment, policy, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]


    #compute_avg_return(eval_env, random_policy, 3)
    initial_collect_steps = 32
    replay_buffer_capacity = 100
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=ppo_agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=replay_buffer_capacity)
    if(0):
        def collect_step(environment, policy):
              time_step = environment.current_time_step()
              action_step = policy.action(time_step)
              next_time_step = environment.step(action_step.action)
              traj = trajectory.from_transition(time_step, action_step, next_time_step)

              # Add trajectory to the replay buffer
              replay_buffer.add_batch(traj)


        for _ in range(initial_collect_steps):
            collect_step(train_env, random_policy)

    if(0):
        action = my_net(time_step.observation, time_step.step_type)
        
        for my_step in range(10):
            time_step = tf_env.step(action)
            action = my_net(time_step.observation, time_step.step_type)

            step = 0

            print('step {}, reward: {}'.format(time_step.step_type, time_step.reward))
        print(time_step)

