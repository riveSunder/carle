import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import carle
from carle.env import CARLE

from carle.mcl import RND2D, AE2D
from carle.agents import RandomAgent, RandomNetworkAgent

def train(agent_fn,\
        instances=16,\
        steps=[64,2048],\
        rules=[[[3],[2,3]]],\
        mcl=[RND2D, AE2D]):
    """
    train endogenous mcl reward wrappers (e.g. RND2D and AE2D)

    Usage:
        instances - sets the vectorization width for CARLE. Default 16
        steps - number of rule change cycles and steps to train per rule change. Default [64, 2^11]
        agent - policy to train with. Default RandomAgent
        rules - a list of lists outling B/S rules for CARLE. [[[B],[S]],[[B],[S]]] 
        mcl - a list of mcl wrappers to apply to CARLE. 
            These are applied sequentially and trained simultaneously

    """


    env = CARLE(instances=instances, use_cuda=True)

    # apply mcl reward wrappers

    env = RND2D(env)
    env = AE2D(env)

    agent = agent_fn(\
            observation_width=env.inner_env.width,\
            observation_height=env.inner_env.height,\
            action_width=env.inner_env.action_width, \
            action_height=env.inner_env.action_height)

    exp_id = "mcl" + str(int(time.time()))

    rewards = []
    t0 = time.time()

    for epoch in range(steps[0]):

        for ruleset in rules:

            env.inner_env.birth = ruleset[0]
            env.inner_env.survive = ruleset[1]

            obs = env.reset()

            sum_reward = 0.0

            t1 = time.time()
            for step in range(steps[1]):

                action = agent(obs)
                obs, reward, done, info = env.step(action)
                sum_reward += reward.detach().sum().cpu().numpy()
                rewards.append(reward.detach().sum().cpu().numpy())

            t2 = time.time()
            steps_per_second = (step * env.inner_env.instances) / (t2-t1)

            print("steps / second = {:.3f}".format(steps_per_second))
            print("round {}, ruleset {}, mean reward = {:.3e}".format(\
                    epoch, ruleset, sum_reward/(step*instances)))
            print("saving mcl state dicts")


            torch.save(env.state_dict(), "./logs/mcl/models/{}_{}.pt".format(\
                    env.my_name, exp_id))

            torch.save(env.env.state_dict(), "./logs/mcl/models/{}_{}.pt".format(\
                    env.env.my_name, exp_id))

        np.save("./logs/mcl/metrics/mcl_rewards_{}".format(exp_id), rewards)


    return rewards

    
if __name__ == "__main__":

    # rules comprised of Life, Move/Morly, Day and Night, and Live Free or Die
    my_rules = [[[3],[2,3]],\
            [[3,6,8],[2,4,5]],\
            [[3,6,7,8],[3,4,6,7,8]],\
            [[3],[0, 2, 3]]]

    my_instances = 8 
    my_agent_fn = RandomAgent
    my_steps = [512, 512]

    rewards = train(my_agent_fn, steps=my_steps, instances=my_instances, rules=my_rules)


    plt.figure()
    plt.plot(rewards)
    plt.show()

