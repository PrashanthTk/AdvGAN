'''
Created on Nov 24, 2017

Purpose is to replicate results of Generating Adversarial Examples with Adversarial Networks, ICLR 2018.
This script implements the full GAN model.
'''

from School.STAT946.Project import Generator, Discriminator, Distilled, Losses
from keras.layers import Input, Add
from keras.models import Model
from keras.optimizers import adam

def Define_GAN(input_shape,alpha,beta):
    G=Generator.Define_Generator(input_shape)
    D=Discriminator.Define_Discriminator(input_shape)
    f=Distilled.Define_Distilled(input_shape)
    
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


