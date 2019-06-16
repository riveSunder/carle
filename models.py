import tensorflow as tf

import numpy as np

import absl 
import absl.flags as flags
import absl.app

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Reshape, AveragePooling1D, Activation

FLAGS = absl.flags.FLAGS

flags.DEFINE_string('model', 'alexnet', 'default model ')


class alexnet(tf.keras.Model):
    def __init__(self, dim_x=32, dim_y=32, dim_out=16):
        super(alexnet, self).__init__(name='alexnet')
        
        #self.conv0 = tf.keras.layers.Conv2D(input_shape=(dim_x,dim_y), activation='relu', padding='same')
        #self.conv1 = tf.keras.layers.Conv2D(activation='relu', padding='same')
        #self.conv2 = tf.keras.layers.Conv2D(actia
        pass

    def call(self, inputs):
        pass

    def get_model(self):
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
        model.add(Conv2D(filters=250, kernel_size=(2,2), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
        model.add(Reshape((16000,1)))
        model.add(AveragePooling1D(pool_size=1000,strides=1000))
        model.add(Reshape(([16])))
        model.add(Activation(tf.nn.softmax))

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
        model.add(Conv2D(filters=250, kernel_size=(2,2), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu, padding='same'))
        model.add(Reshape((16000,1)))
        model.add(AveragePooling1D(pool_size=1000,strides=1000))
        model.add(Reshape(([16])))
        model.add(Activation(tf.nn.softmax))

    model.summary()

if __name__ == '__main__':
    absl.app.run(main)
