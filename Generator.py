'''
Created on Nov 16, 2017

Purpose is to replicate results of Generating Adversarial Examples with Adversarial Networks, ICLR 2018.
This script implements the generator network.
'''


'''
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Convolution2D
from keras_contrib.layers.normalization import InstanceNormalization, BatchRenormalization
'''
    
'''
class Conv_InstNorm_Relu(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Conv_InstNorm_Relu, self).__init__(**kwargs)

    def build(self, input_shape,filters,stride):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Conv_InstNorm_Relu, self).build(input_shape)

    def call(self, x):
        
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)    
    
    
x=Conv_InstNorm_Relu(input_shape=[],)
'''

#https://github.com/PiscesDream/CycleGAN-keras/blob/master/CycleGAN/models/gen/resnet.py

from keras.layers import Convolution2D, ZeroPadding2D,Activation,Add,Convolution2DTranspose
from keras_contrib.layers.normalization import InstanceNormalization

def Conv_InstNorm_Relu(x_input,filters,kernel_size=(3,3),stride=1):
    
    l=ZeroPadding2D()(x_input)
    l=Convolution2D(filters=filters, kernel_size=(3,3), strides=stride,activation='linear')(l)
    l=InstanceNormalization()(l)
    l=Activation('relu')(l)
    
    return l

def Res_Block(x_input, filters, kernel_size=(3,3), stride=1):

    l=ZeroPadding2D()(x_input)
    l = Convolution2D(filters=filters, 
                kernel_size=kernel_size,
                strides=stride,)(l)
    l=InstanceNormalization()(l)
    l = Activation('relu')(l)

    l = ZeroPadding2D()(l)
    l = Convolution2D(filters=filters, 
                kernel_size=kernel_size,
                strides=stride,)(l)
    l = InstanceNormalization()(l)
    merged = Add()([x_input, l])
    return merged
    
def TransConv_InstNorm_Relu(x_input,filters,kernel_size=(3,3),stride=2):
    
    l=Convolution2DTranspose(filters=filters, kernel_size=kernel_size, strides=stride,activation='linear',padding='same')(x_input)
    l=InstanceNormalization()(l)
    l=Activation('relu')(l)
    
    return l

def Define_Generator(input_shape):
    
    from keras.models import Model
    from keras.layers import Input
    
    m_in=Input(shape=input_shape)
    m=Conv_InstNorm_Relu(m_in,filters=8,stride=1)
    
    m=Conv_InstNorm_Relu(m,filters=16,stride=2)
    m=Conv_InstNorm_Relu(m,filters=32,stride=2)
    
    m=Res_Block(m, filters=32)
    m=Res_Block(m, filters=32)
    m=Res_Block(m, filters=32)
    m=Res_Block(m, filters=32)
    
    m=TransConv_InstNorm_Relu(m,filters=16,stride=2)
    m=TransConv_InstNorm_Relu(m,filters=8,stride=2)
    
    m_out=Conv_InstNorm_Relu(m,filters=1,stride=1)
    
    M=Model(m_in,m_out)
    M.compile(optimizer='adam', loss='mean_squared_error')
    
    #M.summary()
    
    return M
    
    
#Define_Generator([28,28,1])    






