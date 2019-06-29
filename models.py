import tensorflow as tf

import numpy as np

import absl 
import absl.flags as flags
import absl.app

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Reshape, AveragePooling1D,\
                                    AveragePooling2D, Activation, Dropout, Flatten, Dense 

FLAGS = absl.flags.FLAGS

flags.DEFINE_string('model', 'alexnet', 'default model ')


class alexnet(tf.keras.Model):
    def __init__(self, dim_x=32, dim_y=32, dim_out=16):
        super(alexnet, self).__init__(name='alexnet')
        
        self.conv0 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, input_shape=(dim_x,dim_y,1), activation='relu', padding='same')
        self.conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, input_shape=(dim_x,dim_y,1), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3,3), strides=1, input_shape=(dim_x,dim_y,1), activation='relu', padding='same')
        pass

    def call(self, inputs):
        pass

    def get_model(self):
        #model = alexnet()
        dim_x, dim_y = 64, 64
        #model.compile(loss='binary_crossentropy',optimizer='adam')
        
        #model = alexnet()
        if(1):#model.compile(loss='binary_crossentropy',optimizer='adam')
            model = Sequential()
            model.add(Conv2D(filters=256, kernel_size=(3,3), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
            model.add(MaxPooling2D(pool_size=2, strides=2))
            model.add(Conv2D(filters=512, kernel_size=(3,3), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
            model.add(MaxPooling2D(pool_size=2, strides=2))
            model.add(Conv2D(filters=1024, kernel_size=(3,3), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
            model.add(SpatialDropout2D(rate=0.125))
            model.add(Conv2D(filters=1024, kernel_size=(1,1), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
            model.add(Conv2D(filters=1024, kernel_size=(2,2), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.tanh, padding='same'))
            #model.add(Reshape((16000,1)))
            #model.add(AveragePooling2D(pool_size=250,strides=250))
            model.add(Flatten())
            model.add(Dropout(rate=0.125))
            #model.add(Dense(256,activation=tf.nn.tanh))
            #model.add(Dense(256,activation=tf.nn.tanh))
            model.add(Dense(16))
            model.add(Reshape(([16])))
            model.add(Activation(tf.nn.sigmoid))

        return model

def main(argv):
    model_name = FLAGS.model

    if model_name == 'alexnet':
        #model = alexnet()
        dim_x, dim_y = 32, 32
        #model.compile(loss='binary_crossentropy',optimizer='adam')
        model = Sequential()
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Conv2D(filters=1024, kernel_size=(3,3), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
        model.add(SpatialDropout2D(rate=0.5))
        model.add(Conv2D(filters=512, kernel_size=(2,2), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
        #model.add(Conv2D(filters=250, kernel_size=(2,2), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
        #model.add(Reshape((16000,1)))
        #model.add(AveragePooling2D(pool_size=250,strides=250))
        model.add(Conv2D(filters=16, kernel_size=(2,2), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
        model.add(Reshape(([dim_x, dim_y,1])))
        model.add(Activation(tf.nn.softmax))

    model.summary()

if __name__ == '__main__':
    absl.app.run(main)
