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
