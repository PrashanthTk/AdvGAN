'''
Created on Nov 22, 2017

Purpose is to replicate results of Generating Adversarial Examples with Adversarial Networks, ICLR 2018.
This script implements the discriminator network.
'''

from keras.layers import Input, ZeroPadding2D, Convolution2D, LeakyReLU, Flatten, Dense
from keras_contrib.layers.normalization import InstanceNormalization
from keras.models import Model


def Define_Discriminator(input_shape):
    m_in=Input(shape=input_shape)
    m=ZeroPadding2D()(m_in)
    m = Convolution2D(filters=8, 
                kernel_size=(4,4),
                strides=2)(m)
    m = LeakyReLU(0.2)(m)
    m=ZeroPadding2D()(m)
    m=   Convolution2D(filters=16, 
                kernel_size=(4,4),
                strides=2)(m)
    m=InstanceNormalization()(m)
    m = LeakyReLU(0.2)(m)
    m=ZeroPadding2D()(m)
    m= Convolution2D(filters=32, 
                kernel_size=(4,4),
                strides=2)(m)
    m=InstanceNormalization()(m)
    m = LeakyReLU(0.2)(m)
    m = Flatten()(m)

    m_out = Dense(1, activation='sigmoid')(m)
    
    M=Model(m_in,m_out)
    M.compile(optimizer='adam', loss='mean_squared_error')
    
    #M.summary()
    
    return M
    
#Define_Discriminator([28,28,1])    