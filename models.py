import tensorflow as tf

import numpy as np

import absl 
import absl.flags as flags
import absl.app

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Reshape, AveragePooling1D,\
                                    AveragePooling2D, Activation, Dropout, Flatten, Dense 

FLAGS = absl.flags.FLAGS

flags.DEFINE_string('model', 'alexnet', 'default model ')


class AlexNet(tf.keras.Model):
    def __init__(self, dim_x=32, dim_y=32, dim_out=16, dropout_rate=0.1, l2_penalty=1e-5):
        super(AlexNet, self).__init__()
        """model based on AlexNet (Krizhevsky et al. 2012)"""
        reg_fn = tf.keras.regularizers.l2(l2_penalty)

        self.dim_x, self.dim_y, self.dim_out = dim_x, dim_y, dim_out

        self.conv0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, \
                input_shape=(dim_x,dim_y,1), activation=tf.nn.leaky_relu, padding='same', kernel_regularizer=reg_fn)
        self.conv1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, \
                input_shape=(dim_x,dim_y,1), activation=tf.nn.leaky_relu, padding='same', kernel_regularizer=reg_fn)
        self.conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, \
                input_shape=(dim_x,dim_y,1), activation=tf.nn.leaky_relu, padding='same', kernel_regularizer=reg_fn)
        self.conv3 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, \
                input_shape=(dim_x,dim_y,1), activation=tf.nn.leaky_relu, padding='same', kernel_regularizer=reg_fn)
        self.conv4 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3,3), strides=1, \
                input_shape=(dim_x,dim_y,1), activation=tf.nn.leaky_relu, padding='same', kernel_regularizer=reg_fn)


        self.do = tf.keras.layers.Dropout(rate=dropout_rate)
        
        self.fc0 = tf.keras.layers.Dense(1024, activation=tf.nn.leaky_relu)
        self.fc1 = tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu)
        self.fc2 = tf.keras.layers.Dense(dim_out, activation='sigmoid')
    
        _ = self.call(np.random.random((1, dim_x, dim_y, 1)))

    def call(self, inputs):
        """
        define model by forward pass
        """
               
        x = self.conv0(inputs) #, input_shape=(self.dim_x, self.dim_y, 1))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

def get_alexnet(dim_x=64, dim_y=64, dim_out=16, dropout_rate=0.1, l2_penalty=1e-5):
    """
    define model by forward pass
    """
    
    reg_fn = tf.keras.regularizers.l2(l2_penalty)

    inputs = tf.keras.layers.Input(shape=(dim_x,dim_y,1))
    x = Conv2D(filters=128, kernel_size=(3,3), strides=1, activation=tf.nn.leaky_relu,\
                padding='same', kernel_regularizer=reg_fn)(inputs)
    x = Conv2D(filters=256, kernel_size=(3,3), strides=1, activation=tf.nn.leaky_relu,\
                padding='same', kernel_regularizer=reg_fn)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=512, kernel_size=(3,3), strides=1, activation=tf.nn.leaky_relu,\
                padding='same', kernel_regularizer=reg_fn)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=1024, kernel_size=(3,3), strides=1, activation=tf.nn.leaky_relu,\
                padding='same', kernel_regularizer=reg_fn)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=2048, kernel_size=(3,3), strides=1, activation=tf.nn.leaky_relu,\
                padding='same', kernel_regularizer=reg_fn)(x)
    
    x = Flatten()(x)
    x = Dense(1024, activation=tf.nn.leaky_relu)(x)
    x = Dense(256, activation=tf.nn.leaky_relu)(x)

    x = Dropout(rate=dropout_rate)(x)
    x = Dense(dim_out, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

def get_encoder():

    pass

def get_decoder():
    """
    decoder of model for predicting automoton rules
    """
    pass

def main(argv):
    model = get_alexnet()
    #model.build(input_shape=(1,dim_x,dim_y,1))
    #model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
    model.summary()

if __name__ == '__main__':
    absl.app.run(main)
