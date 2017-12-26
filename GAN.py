'''
Created on Nov 24, 2017

Purpose is to replicate results of Generating Adversarial Examples with Adversarial Networks, ICLR 2018.
This script implements the full GAN model.
'''

from School.STAT946.Project import Generator, Discriminator, Distilled, Losses
from keras.layers import Input, Add
from keras.models import Model
from keras.optimizers import adam
import keras

def getDistilled_Model(modelname='C'):
    return keras.models.load('./Trained/Untrained_Model'+modelname)

def Define_GAN(input_shape,alpha,beta):
    G=Generator.Define_Generator(input_shape)
    D=Discriminator.Define_Discriminator(input_shape)
    #f=Distilled.Define_Distilled(input_shape)
    modelname='C'
    #the getDistilled function fetches model A or B or C which is pretrained on only pristine MNIST data.
    #File extension not mentioned , by default they are all .h5 files ()
    f=getDistilled_Model('Trained/Model'+modelname+'_PristineOnly')
    x_inp=Input(input_shape)
    perturb=G(x_inp)
    x_perturbed=Add()([x_inp,perturb])
    
    Discrim_Output=D(x_perturbed)
    Class_Output=f(x_perturbed)
    
    GAN=Model(x_inp,[Discrim_Output,Class_Output,perturb])
    GAN.compile(optimizer=adam(lr=0.001),
                loss=[Losses.Custom_MSE,Losses.Adv,Losses.Custom_Hinge(0.3)],
                loss_weights=[1,alpha,beta])
    
    GAN.summary()
    
    return GAN, G, D, f


