'''
Created on Nov 24, 2017

Purpose is to replicate results of Generating Adversarial Examples with Adversarial Networks, ICLR 2018.
This script implements the training process for the GAN model.
'''
import sys
import numpy as np
from School.STAT946.Project import GAN
from keras.datasets import mnist
from keras.utils import to_categorical

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def train_for_n(epochs=5000,batch_size=128):
    from tqdm import tqdm
    
    for _ in tqdm(range(epochs)):
        
        # Make generative images
        real_image_batch = X[np.random.randint(0,X.shape[0],size=int(batch_size/2)),:,:,:]
        fake_image_inp = X[np.random.randint(0,X.shape[0],size=int(batch_size/2)),:,:,:]
        fake_image_batch = np.add(fake_image_inp,G.predict(fake_image_inp))

        # Train discriminator on generated images
        X_batch = np.concatenate((real_image_batch, fake_image_batch))
        y1 = np.zeros([batch_size,1])
        y1[0:int(batch_size/2),] = 1
        
        make_trainable(D,True)
        D.train_on_batch(X_batch,y1)
        #d_loss  = D.train_on_batch(X_batch,y1)
        #losses["d"][e]=d_loss

        #train Generator-Discriminator stack on input noise to non-generated output class
        sample_int=np.random.randint(0,X.shape[0],size=int(batch_size))
        fake_image_inp = X[sample_int,:,:,:]
        y_discrim = np.ones([batch_size,1])
        y_class=y[sample_int]
        y_hinge=np.zeros([batch_size,28,28,1])

        make_trainable(D,False)
        GAN.train_on_batch(fake_image_inp, [y_discrim,y_class,y_hinge])
        #g_loss = GAN.train_on_batch(noise_tr, y2 )
        #losses["g"][e]=g_loss
        
    return

GAN, G, D, f= GAN.Define_GAN([28,28,1], 1, sys.maxsize)

(X,y),(_,_)=mnist.load_data()
X= np.divide(X,255)
X=X.reshape(X.shape[0],28,28,1)
y=to_categorical(y, num_classes=10)

train_for_n(epochs=5000,batch_size=128)