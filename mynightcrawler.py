import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
from tensorflow.python.client import device_lib
import pickle
print(device_lib.list_local_devices())
import numpy as np
#from School.STAT946.Project impimport GAN
import keras
import GAN
from Losses import *
from keras.optimizers import SGD
import keras.backend as K
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = (10000/12000)
set_session(tf.Session(config=config))

#config = tf.ConfigProto()
config.gpu_options.allow_growth=True
# Assume that you have 12GB of GPU memory and want to allocate 500MB:
config.gpu_options.per_process_gpu_memory_fraction=(10000/12000)
sess = tf.Session(config=config)
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

def train_for_n(epochs=5000,batch_size=128):
    #from tqdm import tqdm
    
    for tk in range(epochs):
        if(tk==0):
            print("Entered the loop")
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
        scalarloss=GAN.train_on_batch(fake_image_inp, [y_discrim,y_class,y_hinge])
        #g_loss = GAN.train_on_batch(noise_tr, y2 )
        if(tk%500==0):
            print("Epoch number:",tk,"; Loss",scalarloss)
         #losses["g"][e]=g_loss
    #GAN.save('GAN_CPristine100')
    #G.save('G_CPristine100')
    #D.save('D_CPristine100')
    #f.save('f_CPristine100')
    
    return

#===============================================================================
'''
GAN, G, D, f= GAN.Define_GAN([28,28,1], 100, 10000,'B')
print("=||="*7)
print("=||="*7)
#f.summary()
(X,y),(_,_)=mnist.load_data()
X= np.divide(X,255)
X=X.reshape(X.shape[0],28,28,1)
y=to_categorical(y, num_classes=10)
make_trainable(f,False)
print('Model F\n')
f.summary()
train_for_n(epochs=5000,batch_size=128)
GAN.save_weights('WhiteBox/B_Wbox_Ganwt5k')
G.save_weights('WhiteBox/B_Wbox_Gwt5k')
D.save_weights('WhiteBox/B_Wbox_Dwt5k')
f.save_weights('WhiteBox/B_Wbox_fwt5k')
print('Done')
'''
#=====================================================================================
GAN, G, D, f= GAN.Define_GAN([28,28,1], 100, 10000,'B')
(X,y),(_,_)=mnist.load_data()
X= np.divide(X,255)
X=X.reshape(X.shape[0],28,28,1)
y=to_categorical(y, num_classes=10)

GAN.load_weights('WhiteBox/B_Wbox_Ganwt3k')
G.load_weights('WhiteBox/B_Wbox_Gwt3k')
D.load_weights('WhiteBox/B_Wbox_Dwt3k')
f.load_weights('WhiteBox/B_Wbox_fwt3k')
make_trainable(f,False)

f.summary()
train_for_n(epochs=2000,batch_size=128)
GAN.save_weights('WhiteBox/B_Wbox_Ganwt5k')
G.save_weights('WhiteBox/B_Wbox_Gwt5k')
D.save_weights('WhiteBox/B_Wbox_Dwt5k')
f.save_weights('WhiteBox/B_Wbox_fwt5k')
print('done')