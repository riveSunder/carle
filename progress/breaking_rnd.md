# How to break your Random Network Distillation

Reinforcement learning works great, except when it doesn't, and one of the challenges to learning effective strategies in a reinforcement learning context is dealing with sparse rewards. Choosing problems with a reasonable likelihood of success is part of the art of RL, and this often boils down to looking for heuristics defining good RL problems, <em>e.g.</em>: 

* It should be plausible for an agent acting randomly to stumble upon rewards at every point of the learning curve.
* 
*

A good reference for more in-depth guidelines to approaching problems from an RL perspective is John Schulmann's <a href="https://www.youtube.com/watch?v=8EcdaCk9KaQ">Nuts and Bolts of Deep RL Experimentation</a>

Most RL agents learn by trial-and-error, but this <a href="">can take forever</a>. Adjusting the trade-off between exploitation of learned strategies and exploration of action consequences is a central part of solving even moderately complicated problems with RL. A simple way to do this is to have a parameter like 'temperature' which determines the probability of taking the action with the best predicted value or taking some other possible action. 

More interesting ways to encourage effective exploration attempt to implement curiosity in the learning process. 


include learning uncertainty, predicting, and random network distillation. In this essay we will tinker with great ways to make random network distillation a total waste of effort. There are several self-contained sections which introduce some related background and references. Feel free to skip these if you are familiar with the topic. To spend too much time studying those topics you are already expert in would indicated bad management of your exploration/exploitation trade-off. 

# Reinforcement Learning


# The Environment: Conway's Game of Life


# How to break your RND 



# Strategy 1: recapitulating and learning simple rules

The next state of any given cell in a CA universe is fully determined by the state of its neighbors in an immediately adjacent 3x3 grid. Unsurprisingly, defining an RN as a multilayer conv-net with 3x3 kernels produces an easily predictable distillate. 

# Strategy 2: bad random initialization

weights are too low and outputs are homogenous (all outputs near 0.0 sigmoid to 0.5)

# TODO: Saturate feature outputs

# Strategy 3: non-static random networks
Re-initializing RNs without proper random seed hygeine. (RN resets each episode)


# notes

Normalizing observations for RN and predictor not important in CA environment?

'Hard games' for RL in Atari: Montezuma's revenge, Gravitar, Venture, Solaris, Private Eye, Freeway, and Pitfall! [Bellemare et al. 2016](https://arxiv.org/abs/1606.01868)

### bugs fixed by openai that had big effect on performance:
* 2 and 3, initializing the RN to achieve unsaturated features with a learnable range
* '(our favorite one involved accidentally zeroing an array which resulted in extrinsic returns being treated as non-episodic; we realized this was the case only after being puzzled by the extrinsic value function looking suspiciously periodic)'

sources of high reward signal for next-state prediction-driven learning. 

 Factor 1: Prediction error is high where the predictor fails to generalize from previously seen examples. Novel experience then corresponds to high prediction error.
Factor 2: Prediction error is high because the prediction target is stochastic.
Factor 3: Prediction error is high because information necessary for the prediction is missing, or the model class of predictors is too limited to fit the complexity of the target function.



* For a new algorithm start with some small toy problems
* Run continuous benchmarks against a suite of problems
*

In reinforcement learning agents must strike a fine balance between seeking reward and seeking novelty, the so-called exploitation vs. exploration trade-off. This makes games like Montezuma's Revenge particularly difficult to solve with RL due to the hidden nature of rewards. 

A human player takes quite naturally to this sort of hidden treasure/exploration type game, in part because they can apply prior knowledge from similar games about the connection between keys and doors, for example. 

It's this type of game that drives the need for curiosity driven or intrinsic reward strategies. With this type of approach, the agent seeks to maximize not just the reward from the environment but some proxy for successful exploration.


# RND

Random Network Distillation is an intrinsic reward strategy for reinforcment learning. RND adds two additional models, one trainable, to the 



Of the common video game problems studied as RL problems, Montezuma's revenge is infamously difficult for RL agents. 
 

 They fare particularly poorly on tasks that a human might approach by some combination of curiosity and directed experimentation.  
