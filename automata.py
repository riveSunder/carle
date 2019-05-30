import numpy as np
import matplotlib.pyplot as plt
import sys


class automatons():
    def __init__(self,dim_x=32, dim_y=32, init_prob=0.1,rule='conway'):
        
        self.dim_x = dim_x
        self.dim_y = dim_y
        
        self.plane = np.array(np.random.random((self.dim_x,self.dim_y)) < init_prob,dtype=np.int8)
        self.live_rules = np.zeros((9,)) #1*np.random.random((9,)) < 0.2
        self.dead_rules = np.zeros((9,)) #1*np.array(np.random.random((9,)) < 0.05,dtype=np.int8)
        
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
                
    def propagate(self, plane, steps):
        # propagate the CA universe in plane for a set number of steps

        for ll in range(steps):
            plane = self.step(plane)

        return plane

    def render(self):
        pass
    
def update_ax(ax, new_plane):
    ax.imshow(new_plane, cmap='gray')
    plt.draw()

    plt.pause(0.01)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        rule_name = sys.argv[1]
    else:
        rule_name = 'conway'
    cell = automatons(init_prob=0.25,rule=rule_name)
    
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
        update_ax(ax, cell.plane)
        cell.plane = cell.step(cell.plane)

