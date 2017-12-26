from keras.datasets import mnist
from keras.models import Model, Sequential # basic class for specifying and training a neural network
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras import regularizers
#from keras.preprocessing.image import ImageDataGenerator
#import tensorflow as tf
import keras.backend as K
import time
import keras
import numpy as np
import pickle

import Generator,Discriminator
import layers
x_train,x_test,y_test,y_train=[],[],[],[]
def AdvGAN( generator, discriminator):
	advgan=Sequential()
	advgan.add(generator)

	#Freezing Discriminator Weights
	discriminator.trainable = False
	advgan.add(discriminator)
	return advgan

#Input argument is the model for which you want to generate a distilled version of it: distilled
def get_mnist_data():
	global x_train,x_test,y_test,y_train
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
	pickle.dump(x_train,open('pickles/x_trainm.p',"wb"))
	pickle.dump(x_test,open('pickles/x_testm.p',"wb"))
	pickle.dump(y_train,open('pickles/y_trainm.p',"wb"))
	pickle.dump(y_test,open('pickles/y_testm.p',"wb"))

#loads the normalized version of mnist data
def load_mnist_data():
	global x_train,x_test,y_test,y_train
	x_train=pickle.load(open('x_trainm.p',"rb"))
	y_train=pickle.load(open('y_trainm.p',"rb"))
	x_test=pickle.load(open('y_testm.p',"rb"))
	y_train=pickle.load(open('y_testm.p',"rb"))

#Based on the modelname, it generates a static distilled version of that model ( pristine data only) and returns it
def static_distill_model(model, modelname='C'):
	global x_train,x_test,y_test,y_train
	dist_train_labels=model.predict(x_test)
	#modelresults = np.argmax(prd, axis=1)
	#dist_train_labels=np.equal.outer(modelresults, np.arange(10)).astype(np.float)
	dist_model=''
	if modelname=='A':
		dist_model = Sequential()
		dist_model.add(Conv2D(64, (5,5),padding='valid',
		                input_shape=(28,
		                             28,
		                             1)))
		dist_model.add(Activation('relu'))


		dist_model.add(Conv2D(64, (5,5),padding='valid'))
		dist_model.add(Activation('relu'))

		dist_model.add(Dropout(0.25))

		dist_model.add(Flatten())
		dist_model.add(Dense(128))
		dist_model.add(Activation('relu'))

		dist_model.add(Dropout(0.5))
		dist_model.add(Dense(10))
		dist_model.add(Activation('softmax'))
		opt = keras.optimizers.Adam(lr=0.001)
		dist_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
	if modelname=='B':
		dist_model = Sequential()
		dist_model.add(Conv2D(64, (8,8),padding='valid',
		                        input_shape=(28,
		                                     28,
		                                     1)))
		dist_model.add(Activation('relu'))
		dist_model.add(Dropout(0.2))

		dist_model.add(Conv2D(128, (6,6),padding='valid'))
		dist_model.add(Activation('relu'))
		dist_model.add(Conv2D(128, (5,5),padding='valid'))
		dist_model.add(Activation('relu'))
		dist_model.add(Dropout(0.5))

		dist_model.add(Flatten())
		dist_model.add(Dense(10))
		
		dist_model.add(Activation('softmax'))
 		dist_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

	if modelname=='C':
		model = Sequential()

		model.add(Conv2D(params[0], (3, 3),
		                    input_shape=input_shape))
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


	dist_model.fit(x_train, dist_train_labels, batch_size=batch_size, epochs=6,validation_data=(x_test[0:2000],y_test[0:2000]), verbose=1)

	dist_model.save("staticdist_model"+modelname)
	return dist_model
 
def dynamicdist_modelC(modelname='C'):
	staticmodel=keras.models.load('staticdist_model'+modelname)
	generator=Define_Generator([28,28,1])
	discriminator=Define_Discriminator([28,28,1])
