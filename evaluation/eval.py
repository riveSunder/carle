import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import carle
from carle.env import CARLE

from carle.mcl import RND2D, AE2D, PufferDetector, SpeedDetector
from carle.agents import RandomAgent, RandomNetworkAgent

from evaluation.submission import SubmissionAgent

def evaluate(Agent, rules, wrappers, params_path=None, steps=1024):
    """
    evalution function for CARLE

        Agent - subclassed from DemoAgent, must yield actions from `forward`
        params_path - path to agent parameters to be loaded with \
                `agent._load_state_dict(torch.load(...))`
        rules - a list of lists specifying the B/S CA rules to be evaluated
        wrappers - the mcl reward wrappers, weights for each wrapper,\
                and paths (if any) to load params for them.
        steps - number of params to evaluate each ruleset

    """

    score = 0.0

    agent = Agent()

    if params_path is not None:
        agent.load_state_dict(params_path)

    env = CARLE()


    for wrapper in wrappers:
        env = wrapper[0](env)
        env.reward_scale = wrapper[1]

        env.batch_size = steps*len(rules)

        if wrapper[2] is not None:
            env.load_state_dict(torch.load(wrapper[2]))

        env.eval()

        #env.set_no_grad()

    total_steps = 0
    score_trace = []
    for ruleset in rules:

        env.inner_env.birth = ruleset[0]
        env.inner_env.survive = ruleset[0]

        obs = env.reset()

        for step in range(steps):

            action = agent(obs)

            obs, reward, done, info = env.step(action)

            score +=  reward.detach().sum().cpu().numpy()
            score_trace.append(reward.detach().sum().cpu().numpy())

            total_steps += 1

        print("cumulative score = {:.3e} at total steps = {}, rulset = {}".format(\
                score, total_steps, ruleset))

    score /= total_steps
    
    return score, score_trace

if __name__ == "__main__": 

    wrappers = [\
            [RND2D, 1.0, "evaluation/RND2D_mcl.pt"],\
            [AE2D, 1.0, "evaluation/AE2D_mcl.pt"],\
            [SpeedDetector, 1e-2, None], \
            [PufferDetector, 1e-3, None]]
    
    my_rules = [\
            [[3,6,8],[2,4,5]],\
            [[3], [2,3]],\
            [[3,6,7,8],[3,4,6,7,8]],\
            [[3],[0,2,3]],\
            [[2],[0]]]
    
    Agent = SubmissionAgent
    
    score, score_trace = evaluate(Agent, my_rules, wrappers, params_path=None, steps=1024)


    if(1):
        print("mean evaluation score is {:.3e}".format(score))

        plt.figure()
        plt.plot(score_trace, 'g', lw=3)
        plt.title("Random agent baseline")
        plt.xlabel("steps, (final ruleset is outgroup)")
        plt.show()


