import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D,Conv2D, BatchNormalization, Flatten, Dense, Input, concatenate,Conv3D,MaxPooling3D,SpatialDropout1D
from keras import regularizers
import keras.backend as K
from keras.layers import Activation, Lambda
import numpy as np


class TCN:
    def __init__(self,nb_filters,**kwargs):
        self.activation = kwargs['activation']
        self.nb_stacks = 1
        self.dilations = None
        self.kernel_size = 2
        self.padding = 'causal'
        self.nb_filters = nb_filters
        self.return_sequences = True
        self.dropout_rate = 0.0
        self.use_skip_connections = True
        self.nb_stacks = 1
        self.name = 'tcn'

        if self.padding != 'causal' and self.padding != 'same':
            raise ValueError("Only 'causal' or 'same' paddings are compatible for this layer.")
    
    def relu_activation(x): ##RELU Activation
        x =  Activation('relu')(x)  
        max_values = K.max(K.abs(x),2,keepdims=True) + 1e-5
        out = x/max_values
        return out
        
    def wave_net_activation(x): ##WAVENET Activation
        tanh_out = Activation('tanh')(x)
        sigm_out = Activation('sigmoid')(x)
        return tf.keras.layers.multiply([tanh_out,sigm_out])
    
    def residual_block(self,x,s,i,name):
        input_x = x
        conv = Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size, dilation_rate=i,padding=self.padding, name= name+'_dilated_conv_%d_tanh_s%d' % (i, s))(input_x)
        if (self.activation == 'wavenet'):
            x = self.wave_net_activation(conv)
        elif(self.activation == 'relu_activation'):
            x = self.relu_activation(conv)
        x = SpatialDropout1D(dropout_rate=self.dropout_rate, name= name+'_spatial_dropout1d_%d_s%d_%f' % (i, s, 0.0))(x)
        x = Conv1D(self.nb_filters, 1, padding='same')(x)
        res_x = tf.keras.layers.add([input_x, x])
        return res_x, x
    
    def tcn_model(self,inputs,**kwargs):
        self.activation = kwargs['activation']
        self.dilations = [1,2,4,8,16,32]
        x = inputs
        x = Conv1D(self.nb_filters, 1, padding=self.padding, name=self.name+'initial_conv')(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for i in self.dilations:
                x,skip_out = self.residual_block(x,s,i,self.name)
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            x = tf.keras.layers.add(skip_connections)
        x = Activation('relu')(x)
        return x