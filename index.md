---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---


# Carle's Game: A Challenge in Open-Ended Simplicity, Complexity, and Creative Exploration

Cellular Automata Reinforcement Learning Environment (CARLE) is a flexible, fast environment for training creative exploration in machine intelligence. 

## Open-Endedness

While CARLE is formulated as a reinforcement learning environment and uses the the now-standard OpenAI gym pattern  (`obs, reward, done, info = env.step(action)`), `done` is never returned as `True` and the environment itself only every returns a reward of 0.0. Users of CARLE are encouraged to develop their own strategies to instill training with a sense of curiosity, planning, play, and anything else that might encourage machine creativity.

As an example, CARLE includes an environment wrapper that implements random network distillation (RND, [Burda et al. 2018](https://arxiv.org/abs/1810.12894)) as a curiosity reward. The following animations demonstrate RND rewards for a "toggle every toggle" policy and a random policy that toggles 2% of cells for Conway's Game of life (B3/S23) and "Mouse Maze" (B37/S12345). 

<div align="center">
<img src="/carle/assets/rnd_ones_life.gif">
<br>
Conway's Game of Life with an agent that toggles every cell in the action space at every time step. 
<br>
<img src="/carle/assets/rnd_ones_mouse.gif">
<br>
"Mouse Mazze" CA with an agent that toggles every cell in the action space at every time step. 
<br>

<img src="/carle/assets/rnd_random_life.gif">
<br>
Conway's Game of Life with an agent that randomly toggles 2% cells in the action space at every time step. 
<br>

<img src="/carle/assets/rnd_random_mouse.gif">
<br>
"Mouse Maze" CA with an agent that randomly toggles 2% cells in the action space at every time step. 
<br>

</div>

The examples demonstrate that RND likes complex chaos, whether or not that complexity corresponds to what humans would consider interesting machines. Also, curiosity driven rewards can be expected to be sensitive to each specific set of CA rules.Carle's Game will encourage participants to come up with a flexible scheme for encouraging creative exploration across a number of different CA rules that may have vastly different characteristics. A more contrived example, shown below, demonstrates RND getting "bored" by a [Gosper Glider Gun](https://www.conwaylife.com/wiki/Gosper_glider_gun), until gliders wrap around the universe and collide with the machine, creating exciting chaos. 

<div align="center">
<img src="/carle/assets/rnd_experiments/gosper_glider_surprise.gif">
<br>
<img src="/carle/assets/rnd_experiments/screen_tb_gosper_surprise_reward.png">
</div>

## Flexibility

CARLE offers the flexibility to run cellular automata under [Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)-like rule sets. CA universes consist of 2D grids of cells with binary states of 1 or 0, typically termed "alive" or "dead." Cell updates are defined in a set of conditions for cells to spontaneously convert from dead to living states ("birth") and to persist if they are already alive. These rules are based on the sum of states of adjacent cells in a [Moore neighborhood](https://en.wikipedia.org/wiki/Moore_neighborhood). The sum of cell states in a 2D Moore neighborhood can take on 9 values from 0 to 8, therefore Life-like birth and survival rules occupy a space of possible rulesets with 2^9 * 2^9 or 262,144 possible configurations. Many of these rules create complex and chaotic evolution and the possibility for building beutiful machines (including [Turing complete computers](http://www.rendell-attic.org/gol/tm.htm)).

In addition to John Conway's Game of Life, CARLE supports Life-like cellular automata including "High Life," "Amoeba," "Mouse Maze," and many others. Arbitrary CA birth/survival (aka B/S) rules can be used for a high level of diversity. In additional, an official competition will use a different test environment that may use a different neighborhood and update rule formulation, although the observation and action spaces will remain the same 2D CA grid universe. This ensures a challenge that requires agents to have creative exploration and meta-learning at test time.

## Speed

CARLE implements Life-like cellular automata universes using the deep learning library PyTorch and takes advantage of environment vectorization. As a result, the environmnet can run at a rate of thousands of updates per second on a consumer laptop with 2 cores. A fast implementation lowers the barriers to making meaningful contributions to machine creativity and also increases the chances of actually developing creative agents in the first place.

| Vectorization Factor | Updates per Second (GoL) |
|----------------------|--------------------------|
| 1 | 2239.77 |
| 2 | 3663.11 |
| 4 | 5618.96 |
| 8 | 7326.30 |
| 16 | 8779.06 |
| 32 | 9378.68 |
| 64 | 9657.42 |
| 128 | 9682.31 |
| 256 | 7581.44 | 

<div align="center"><em>
Number of updates per second for 64x64 CA grid universes running under Conway's Game of Life rules at various levels of parallelization. This example was run on a laptop with an Intel i5 CPU at 2.40GHz with 2 cores, 4 threads.
</em></div>


| Vectorization Factor | Updates per Second (GoL) |
|----------------------|--------------------------|
| 1 | 348.59 |
| 2 | 616.23 |
| 4 | 959.89 |
| 8 | 1322.34 |
| 16 | 1710.91 |
| 32 | 1966.00 |
| 64 | 2096.01 |
| 128 | 1885.97 |
| 256 | 1814.00 | 

<div align="center"><em>
Number of updates per second for 64x64 CA grid universes running under Conway's Game of Life rules with a random network distillation ([Burda et al. 2018](https://arxiv.org/abs/1810.12894)) exploration bonus wrapper, and at various levels of parallelization. This example was run on a laptop with an Intel i5 CPU at 2.40GHz with 2 cores, 4 threads.
</em></div>

## Environment Description

<em>This description of CARLE may change</em>

CARLE is formulated as a reinforcement learning environment for binary CA grid universes. Although it follows RL patterns, it doesn't provide a non-zero reward of done signal on it's own, that task is left as an exercise to participants. 

The action space is 32x32 and is expected to take binary values: 1 to toggle the cell at each location and 0 to leave it alone. The observation space is nominally 64x64 but can be adjusted by changing `env.height` and `env.width`, and may be changed in the final training and/or test environments. Observations are provided as Nx1xHxW tensors, and actions are expected in the same format. Notably, N can be greater than 1 when training and this allows for many CA universes to be updated in parallel, this is adjusted by changing the `env.instances` variable.  

CA grids represent the surface of a toroid, _i.e._ the unverse wraps around on itself like Pac-Man. 

## Timeline

I'm currently looking for a conference to host CARLE as an official competition. The timeline for participation in the contest will ultimately depend in part 

* 2021 March: Competition Launch
* 2021 March to May: Beta Round. In this round participants are encouraged to experiment with CARLE, raise issues, provide feedback, and make comments that help make CARLE and the associated challenge more interesting.
* 2021 May to Mid-July: Round 1 will have machine agents and their caretakers attempting to maximize open-ended metrics on 4 to 8 known CA rulesets. Metrics may include, but won't be limited to, such accomplishments as duration of aperiodic dynamism without toggling cells in the action space, center of mass displacement speed without toggling cells, or human preferences. 
* 2021 Late July: Final submissions due, test round begins. Machine agents (on their own this time) will be presented with previously unseen CA rules to see what they come up with.  
* 2021 August: Final evaluations, fabulous prizes, prestige, presentation, etc. 
