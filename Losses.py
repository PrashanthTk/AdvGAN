'''
Created on Nov 24, 2017

Purpose is to replicate results of Generating Adversarial Examples with Adversarial Networks, ICLR 2018.
This script codes the custom loss functions.
'''

from keras import backend as K

def Custom_MSE(y_true,y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def Adv(y_true,y_pred):
    real=K.sum(y_true*y_pred,axis=-1)
    other=K.max((1-y_true)*y_pred-y_true*10000, axis=-1)
    return K.maximum(real-other, 0)

def Custom_Hinge(c):
    def loss(y_true,y_pred):
        print(y_true)
        return K.maximum(K.max(K.abs(y_pred),axis=[-2,-1])-c,0)
    return loss


