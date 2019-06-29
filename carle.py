import numpy as np
import matplotlib.pyplot as plt
import sys
import io
import time 

import absl
import absl.app
import absl.flags

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from models import alexnet
#FLAGS = absl.FLAGS

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Reshape, \
                                    AveragePooling1D, AveragePooling2D, Activation, \
                                    Dropout, Dense, Flatten 


class cellular_automata():
    def __init__(self, dim_x=64, dim_y=64, dim_distillate=16, init_prob=0.1, rule='conway'):
        # universe dimensions    
        self.dim_x = dim_x
        self.dim_y = dim_y
       
        #
        self.dim_distillate = dim_distillate
        #reward weighting
        self.w_rnd = 1.0
        self.w_der = 1.0 - self.w_rnd

        
        self.reset(init_prob,rule)
        self.distiller = None
        
        self.plane_memory = np.zeros_like(self.plane)

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

    def rl_step(self, plane, action, prediction, steps=1):
        """toggle the cells specified in action and advance the environment by (default) one step
        plane - current state of ca universe
        action - array with same dimensions as plane designating which cell states to toggle
        prediction - the predicted random network distillate after the next step
        steps - number of steps to propagate (added for flexibility, not used yet)
        """
        
        # initialize returnables
        new_plane = np.zeros_like(plane)
        reward = 0.0
        distillate = np.zeros_like(prediction) 
        info = {'done': False}
        
        
        # toggle the toggles (XOR plane and action)
        action = np.array(action, dtype=np.uint8)
        plane = (plane | action) - (plane & action)

        # step and distill
        new_plane = self.step(plane)
        distillate = self.distill(plane)
        self.plane = plane

        # compute reward 
        rnd_reward = np.sum(np.abs(prediction - distillate))

       
        derivative_reward0 = np.sum(np.abs(new_plane - plane))
        derivative_reward1 = np.sum(np.abs(new_plane - self.plane_memory))
        derivative_reward = (derivative_reward0 + derivative_reward1)/2
        self.plane_memory = plane

        reward = self.w_rnd * rnd_reward + self.w_der * derivative_reward

        if derivative_reward0 == 0 or derivative_reward1 == 0:
            info['done'] = True

        return new_plane, reward, distillate,  info

    def distill(self, plane, random_seed=29):
        """random network distillation of the input plane"""
        
        # get universe dimensions
        dim_x, dim_y = plane.shape[0], plane.shape[1]
        # hidden layer dimension
        dim_h = 128
        dim_out = 16

        if self.distiller is None:
            # generatie distillation reservoir if none exists
            current_state = np.random.get_state()
            np.random.seed(random_seed)
            
            if(0):

                self.distiller = {}
                self.distiller['w0'] = 3e-1*np.random.randn(dim_x*dim_y, dim_h)
                self.distiller['w1'] = 3e-1*np.random.randn(dim_h, dim_out)
            np.random.set_state(current_state)
            
            self.distiller = Sequential()
            self.distiller.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
            self.distiller.add(MaxPooling2D(pool_size=4, strides=4))
            self.distiller.add(Conv2D(filters=64, kernel_size=(3,3), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
            self.distiller.add(MaxPooling2D(pool_size=2, strides=2))
            self.distiller.add(Conv2D(filters=128, kernel_size=(3,3), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
            self.distiller.add(MaxPooling2D(pool_size=2, strides=2))
            self.distiller.add(Conv2D(filters=256, kernel_size=(2,2), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
            self.distiller.add(MaxPooling2D(pool_size=2, strides=2))
            self.distiller.add(Conv2D(filters=128, kernel_size=(1,1), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
            self.distiller.add(MaxPooling2D(pool_size=2, strides=2))
            self.distiller.add(Conv2D(filters=16, kernel_size=(1,1), strides=1,\
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=2.5),\
                    input_shape=(dim_x,dim_y,1), padding='same'))
            self.distiller.add(Flatten())
            self.distiller.add(Dense(16,kernel_initializer=tf.keras.initializers.Identity))
            #self.distiller.add(Dense(self.dim_distillate,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=random_seed)))
            self.distiller.add(Activation(tf.nn.sigmoid))
            self.distiller.add(Reshape([16]))
            self.distiller.trainable = False
            #self.distiller.summary()
        # run plane through the random network
        if(len(plane.shape) == 3):
            # if plane is in (samples, x, y) format, reshape to (samples, x*y), else reshape to (sample, x, y)
            x = plane.reshape(plane.shape[0], plane.shape[1],plane.shape[2],1)
        else:
            x = plane.reshape(1, plane.shape[0],plane.shape[1],1)
       
        x = self.distiller.predict(x)  
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

def get_fig(new_plane, predicted_distillate, distillate):
    """
    Return a figure comparing the reconstruction and associated uncertainty to the target and error
    """
    # get a uniform max/min for displaying confidence and error
    #fig_min = np.min([np.min(var), np.min(np.abs(real-recon))])
    #fig_max = np.max([np.max(var), np.max(np.abs(real-recon))])
    dim_x, dim_y = new_plane.shape[1], new_plane.shape[2]
    
    figure = plt.figure(figsize=(12,5))
    plt.subplot(131)
    plt.imshow(new_plane.reshape(dim_x,dim_y), cmap='magma')
    plt.subplot(132)
    plt.imshow(predicted_distillate.reshape(4,4), cmap='magma')
    plt.colorbar()
    plt.title('predicted')
    plt.subplot(133)
    plt.imshow(distillate.reshape(4,4), cmap='magma')
    plt.title('distillate')
    plt.colorbar()
    return figure

def plot_to_image(figure):
    """
    This function comes from the example at https://www.tensorflow.org/tensorboard/r2/image_summaries
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def step_test():

    #tf.debugging.set_log_device_placement(True)

    #ax = plt.axes()
    toggle_prob = 0.00

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        logical_gpus =  tf.config.experimental.list_physical_devices('GPU')
        print(len(gpus), 'phys gpus', len(logical_gpus), 'logical gpu')


    alex = alexnet()
    predictor = alex.get_model()
    predictor.compile(loss='mse', optimizer=Adam(lr=1e-4),metrics=['acc'])
   
    for seed in range(10):
        tf.random.set_seed(seed)

        unique_id = hash(time.time())
        train_summary_writer = tf.summary.create_file_writer('./logs/seed{}_{}'.format(seed,unique_id))
       
        # maximum number of episodes to train for
        max_episodes = 900
        # maximum number of steps per episodes
        max_steps = 16

        summary_count = 0
        for episode in range(max_episodes):
            # reset the ca universe and 'done' state
            ca = cellular_automata(init_prob=0.15, rule='conway')
            plane = ca.plane
            step = 0
            info = {'done': False}

            while(info['done'] == False):

                action = np.random.random(size=(ca.plane.shape[-2], ca.plane.shape[-1]))
                
                action[action < (1-toggle_prob)] = 0.
                action[action >= (1-toggle_prob)] = 1.

                if(0):
                    prediction = np.random.randn(1,16)
                else:
                    x = plane.reshape(1, plane.shape[-2], plane.shape[-1], 1)
                    prediction = predictor.predict(x)

                
                action = np.array(action, dtype=np.uint8)
                old_plane =  ((plane | action) - (plane & action)).reshape(1,plane.shape[-2], plane.shape[-1],1)
                plane, reward, distillate, info = ca.rl_step(plane, action, prediction)
                #print('episode {} step {} reward: {}'.format(episode, step, reward))
                 
                new_plane = plane.reshape(1, plane.shape[-2], plane.shape[-1],1)

                try: 

                    new_planes = np.apped(new_planes, new_plane, axis=0) 
                    old_planes = np.append(old_planes, old_plane, axis=0)
                    distillates = np.append(distillates, distillate, axis=0)
                    if distillates.shape[0] > 30000:
                        distillates = distillates[:30000,...]

                        old_planes = old_planes[:30000,...]
                        new_planes = new_planes[:30000,...]
                except:
                    new_planes = new_plane
                    old_planes = old_plane
                    distillates = distillate
        
        
                if(1):
                    start_loss = predictor.evaluate(new_planes, distillates, verbose=0)
                    predictor.fit(new_planes, distillates, batch_size=64, epochs=3, verbose=0)
                    end_loss = predictor.evaluate(new_planes, distillates, verbose=0)
                else:
                    start_loss = predictor.evaluate(old_planes, new_planes, verbose=0)
                    predictor.fit(old_planes, new_planes, batch_size=64, epochs=32, verbose=0)
                    end_loss = predictor.evaluate(old_planes, new_planes, verbose=0)

                if(step % 8 == 0):
                    
                    print('episode {} step {} reward: {}, loss: {}'.format(episode, step, reward, end_loss))
                    with train_summary_writer.as_default():
                        predicted_distillate = predictor.predict(new_plane)#.reshape(4,4)
                        tf.summary.scalar('prediction loss', end_loss[0], step=summary_count)
                        tf.summary.scalar('accuracy', end_loss[1], step=summary_count)
                        tf.summary.scalar('reward', reward, step=summary_count)
                        my_fig = get_fig(new_plane, predicted_distillate, distillate)
                        tf.summary.image('distillate_comp', plot_to_image(my_fig), step=summary_count)
                        summary_count += 1
                        #tf.summary.image('distillate', distillate.reshape(1,4,4,1), step=summary_count)
                        #tf.summary.image('predicted_distillate', predicted_distillate.reshape(1,4,4,1), step=summary_count)
                        #tf.summary.image('ca state', tf.Variable(new_plane), step=summary_count)
                    
                if(0):
                    plt.figure(figsize=(12,5))
                    plt.subplot(131)
                    plt.imshow(new_plane.reshape(32,32), cmap='magma')
                    plt.subplot(132)
                    plt.imshow(predicted_distillate, cmap='magma')
                    plt.title('predicted')
                    plt.subplot(133)
                    plt.imshow(distillate.reshape(4,4), cmap='magma')
                    plt.title('distillate')
                    plt.show()

                #im = ax.imshow(ca.plane, cmap='gray')
                
                #ca.render(im, ca.plane)
                #ax.cla()
                step += 1
                if step > max_steps: info['done'] = True


def main(argv):
    #ca = cellular_automata(init_prob=0.25, rule='conway')
    #print(ca.distill(ca.plane))
    step_test()

if __name__ == '__main__':
    absl.app.run(main)
