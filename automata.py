import numpy as np
import matplotlib.pyplot as plt
import sys
import time 

import absl
import absl.app
import absl.flags

#FLAGS = absl.FLAGS



class cellular_automata():
    def __init__(self,dim_x=32, dim_y=32, init_prob=0.1,rule='conway'):
        # universe dimensions    
        self.dim_x = dim_x
        self.dim_y = dim_y
        
        self.reset(init_prob,rule)
        self.distiller = None
        
    def reset(self, init_prob, rule):
        
        # random slate
        self.plane = np.array(np.random.random((self.dim_x,self.dim_y)) < init_prob,dtype=np.int8)
        
        #self.fig, self.ax = plt.subplots(1,1,figsize=(8,8))



        #determine rules
        self.live_rules = np.zeros((9,)) 
        self.dead_rules = np.zeros((9,)) 
        
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
        
    def step(self,plane):
    
        sum_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])

        new_plane = np.copy(plane)
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
        self.plane = new_plane
        return new_plane

    def rl_step(self, plane, action):
        """"""


        pass

    def distill(self, plane, random_seed=29):
        """random network distillation of the input plane"""
        
        # get universe
        dim_x, dim_y = plane.shape[0], plane.shape[1]
        # hidden layer dimension
        dim_h = 128
        dim_out = 1

        if self.distiller is None:
            # generatie distillation reservoir if none exists
            current_state = np.random.get_state()
            np.random.seed(random_seed)

            self.distiller = {}
            self.distiller['w0'] = np.random.randn(dim_x*dim_y, dim_h)
            self.distiller['w1'] = np.random.randn(dim_h, dim_out)

        # run plane through the random network
        if(len(plane.shape) == 3):
            # if plane is in (samples, x, y) format, reshape to (samples, x*y), else reshape to (sample, x, y)
            x = plane.reshape(plane.shape[0], plane.shape[1]*plane.shape[2])
        else:
            x = plane.reshape(1, plane.shape[0]*plane.shape[1])
        for name in self.distiller:
            # Sequential random network 
            x = self.relu(x)
            x = np.matmul(x, self.distiller[name])
                
        return x

    def relu(self, z):
        """return the relu of z"""

        return np.max([np.zeros_like(z),z], axis=0)

    def propagate(self, plane, steps):
        # propagate the CA universe in plane for a set number of steps

        for ll in range(steps):
            plane, reward, info = self.step(plane)

        return plane

    def render(self, im, plane, doblit=False):
        plt.ion()

        #while True:
        im.set_data(plane)
        plt.pause(0.025)

        plt.ioff() # due to infinite loop, this gets never called.

def update_ax(ax, new_plane):
    ax.imshow(new_plane, cmap='gray')
    plt.draw()

    plt.pause(0.01)

def animate_ca():
    if len(sys.argv) > 1:
        rule_name = sys.argv[1]
    else:
        rule_name = 'conway'
    cell = cellular_automata(init_prob=0.25,rule=rule_name)
    
    if len(sys.argv)>2:
        cell.plane *= 0
        glider = np.array([[1, 1, 1],[0,0,1],[0,1,0]],dtype=np.int8)
        cell.plane[20:23,20:23] = glider

    fig, ax = plt.subplots(1,1,figsize=(8,8))

    ax = plt.axes()

    for dd in range(100):

        #plt.figure(figsize=(8,8))
        #plt.imshow(cell.plane,cmap='gray')
        #plt.show
        #update_ax(ax, cell.plane)
        im = ax.imshow(cell.plane, cmap='gray')
        cell.render(im, cell.plane)
        cell.plane = cell.step(cell.plane)
        ax.cla()

def main(argv):
    ca = cellular_automata(init_prob=0.25, rule='conway')
    print(ca.distill(ca.plane))

if __name__ == '__main__':
    absl.app.run(main)
