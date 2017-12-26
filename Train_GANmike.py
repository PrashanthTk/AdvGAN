'''
Created on Nov 24, 2017

Purpose is to replicate results of Generating Adversarial Examples with Adversarial Networks, ICLR 2018.
This script implements the training process for the GAN model.
'''

import numpy as np
#from School.STAT946.Project import GAN, Distill_Model
import GAN, Distill_Model
#from School.STAT946.Project.mnist_challenge.model import Model as targ_model
from keras.datasets import mnist
from keras.utils import to_categorical
from tqdm import tqdm
from keras.models import load_model as load
import tensorflow as tf
from sklearn.metrics import accuracy_score
#from matplotlib import pyplot as plt

#system_path= 'ResearchDocuments/STAT946/Project/'
system_path= './'

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def train_for_n(batches=20000,batch_size=128,distillation='static'):

    #Distill the adversarial model
    print('Initial distillation (same for static and dynamic)')
    Distill_Model.Distill_Model(target=target, model=f, X=X,iterations=5, mode='epoch')


    for t in range(batches):

        # Make generative images & select real images
        real_image_batch = X[np.random.randint(0,X.shape[0],size=int(batch_size/2)),:,:,:]
        fake_sample=np.random.randint(0,X.shape[0],size=int(batch_size/2))
        fake_image_inp = X[fake_sample,:,:,:]
        fake_image_batch = np.add(fake_image_inp,G.predict(fake_image_inp))

        # Train discriminator on generated images
        X_batch = np.concatenate((real_image_batch, fake_image_batch))
        y1 = np.zeros([batch_size,1])
        y1[0:int(batch_size/2),] = 1

        #train the discriminator
        make_trainable(D,True)
        D.train_on_batch(X_batch,y1)

        #sample more real images to train the generator
        sample_int=np.random.randint(0,X.shape[0],size=int(batch_size))
        fake_image_inp = X[sample_int,:,:,:]
        y_discrim = np.ones([batch_size,1])
        y_class=y[sample_int]
        y_hinge=np.zeros([batch_size,28,28,1])

        make_trainable(D,False)
        make_trainable(f,False)
        h_GAN = GAN_mod.train_on_batch(fake_image_inp, [y_discrim,y_class,y_hinge])

        #Dynamic distillation
        if distillation=='dynamic':
            make_trainable(f,True)
            Distill_Model.Distill_Model(target=target, model=f, X=X_batch,iterations=2,mode='batch')

        if t%100==0:
            targ_pred=np.argmax(target.predict(fake_image_batch),1)
            adv_acc=accuracy_score(np.argmax(y[fake_sample],1),targ_pred)
            print('Accuracy: ' + str(adv_acc))
            #plt.imshow(fake_img_batch[0,])

    return


GAN_mod, G, D, f= GAN.Define_GAN([28,28,1], 1, np.infty)

print('Importing target model')
def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,logits=predicted)
target=load('./ModelC',custom_objects={'fn': fn})


print('Loading MNIST data')
(X,y),(X_test,y_test)=mnist.load_data()
X= np.divide(X,255)
X=X.reshape(X.shape[0],28,28,1)
X_test= np.divide(X_test,255)
X_test=X_test.reshape(X_test.shape[0],28,28,1)
y=to_categorical(y, num_classes=10)
y_test=to_categorical(y_test, num_classes=10)

print('Training')
train_for_n(batches=20000,batch_size=128,distillation='dynamic')

print('Saving models')
GAN_mod.save_weights(system_path + 'ModelC_AdvGAN_dynamicdist.hdf5')
f.save_weights(system_path + 'ModelC_Distilled.hdf5')

#GAN_mod.load_weights(system_path + 'ModelC_AdvGAN_dynamicdist.hdf5')

X_test_perturb=np.add(X_test,G.predict(X_test))
targ_pred=np.argmax(target.predict(X_test_perturb),1)
adv_acc=accuracy_score(np.argmax(y_test,1),targ_pred)
print('Test Accuracy: ' + str(adv_acc))
