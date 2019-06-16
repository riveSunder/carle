import numpy as np
import matplotlib.pyplot as plt
import sys
import time 

from automata import cellular_automata

dataset_size = 4096
random_seed = 42
max_steps = 250
display_every = 128
np.random.seed(random_seed)

if __name__ == '__main__':

    rules = ['conway', 'maze', 'coral', 'gnarl',\
            'amoeba', 'walled_cities', 'day_and_night']
    print(len(rules))
    ca = cellular_automata(rule='conway')
    ii = 0

    t0 = time.time()
    while(ii < dataset_size):
        
        if ii % display_every == 0: 
            print('step {}, {} s elapsed'.format(ii,time.time()-t0))
        
        target = np.array(np.random.randint(len(rules)))
        ca.reset(init_prob=0.25, rule=rules[target])
        new_plane = ca.propagate(ca.plane, np.random.randint(max_steps))

        # don't keep dead universes
        if np.sum(new_plane > 0):
            try:
                x = np.append(x, new_plane.reshape(1, ca.dim_x, ca.dim_y, 1), axis=0)
                y = np.append(y, target.reshape(1,1), axis=0)
            except:
                x = new_plane.reshape(1,ca.dim_x,ca.dim_y,1) 
                y = target.reshape(1,1)
            
            ii += 1

    np.save('./data/train_x500.npy',x)
    np.save('./data/train_y500.npy',y)
