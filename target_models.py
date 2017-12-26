'''
Created on Nov 24, 2017

Purpose is to replicate results of Generating Adversarial Examples with Adversarial Networks, ICLR 2018.
This script defines and trains target models: A,B, and C.
'''

# train_models.py -- train the neural network models for attacking
#
# Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
#
# This program is licenced under the BSD 2-Clause licence,
# contained in the LICENCE file in this directory.
from keras.datasets import mnist
from keras.models import Model, Sequential # basic class for specifying and training a neural network
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras import regularizers

import keras.backend as K
import keras
import numpy as np
import pickle
from keras.optimizers import Adam

import tensorflow as tf
np.random.seed(2017)
# MNIST Data Pre-processing 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],
                          28,
                          28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train, 10).astype(np.float32)
y_test = np_utils.to_categorical(y_test, 10).astype(np.float32)

def target_model_train(model_name, nepochs, batch_size):
    if model_name=='A':
        model = Sequential()
        model.add(Conv2D(64, (5,5),padding='valid',
                input_shape=(28,
                             28,
                             1)))
        model.add(Activation('relu'))


        model.add(Conv2D(64, (5,5),padding='valid'))
        model.add(Activation('relu'))

        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        opt = keras.optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    if model_name=='B':
        model = Sequential()
        model.add(Conv2D(64, (8,8),padding='valid',
                        input_shape=(28,
                                     28,1)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (6,6),padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (5,5),padding='valid'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(10))

        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    if(model_name=='C'):
        
        params=[32, 32, 64, 64, 200, 200]
    
        model = Sequential()
    
        model.add(Conv2D(params[0], (3, 3),
                            input_shape=(28,
                                     28,1)))
        model.add(Activation('relu'))
        model.add(Conv2D(params[1], (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(params[2], (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(params[3], (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(params[4]))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(params[5]))
        model.add(Activation('relu'))
        model.add(Dense(10,activation='softmax'))
    
        def fn(correct, predicted):
            return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

        adam = Adam(lr=0.001)
    
        model.compile(loss=fn,
                  optimizer=adam)
    model.fit(x_train, y_train,batch_size=batch_size,
              epochs=nepochs)        
    return model

target_model=target_model_train("C",50,128)
prd = target_model.predict(x_test)
prd_y = np.argmax(prd, axis=1)
y_test = np.argmax(y_test, axis=1)

nb_correct_labels = np.sum(prd_y == y_test)
print('Test accuracy is: ', nb_correct_labels/len(y_test))
target_model.save('./ModelC.h5') # creates a HDF5 file 'mlp1.h5'
#results=target_model.evaluate(x_test,y_test,verbose=1)
#print('Test accuracy:', results)
