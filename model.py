# -*- coding: utf-8 -*-
"""
Created on Sat Nov 07 00:31:47 2021

@author: bharathi
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D,Conv2D, BatchNormalization, MaxPooling2D,Flatten, Dense, Input, concatenate,Conv3D,MaxPooling3D,SpatialDropout1D
from keras import regularizers
import keras.backend as K
from keras.layers import Activation, Lambda
import numpy as np
import tcn_
class Convolution:
        
    def Archi_3DCONV(X,nb_classes,multi_task):
        l2_rate = 1.e-6
        kernel_regularizer = regularizers.l2(l2_rate)
        kernel_initializer = "he_normal"
        input_shape = X.shape
        ##parameters of architecture
        l2_rate = 1.e-6
        dropout_rate = 0.5
        nb_conv = 3
        nb_fc =1
        nbunits_conv = 64
        nbunits_fc = 256
        X_input = Input(input_shape)
        
        X = X_input
        X = Conv3D(kernel_size=(3,3,7),filters = 4, activation = 'relu',kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(X)
        print(X.shape)
        X = MaxPooling3D(pool_size=(2,2,2))(X)
        X = Conv3D(filters=2, kernel_size=(3,3,3), activation='relu',kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(X)
        X = MaxPooling3D(pool_size=2)(X)
        #X = Conv3D(filters=4, kernel_size=(3,3,3), activation='relu')(X)
        #X = MaxPooling3D(pool_size=2)(X)
        #X = Conv3D(filters=4, kernel_size=(3,3,3), activation='relu')(X)
        #X = MaxPooling3D(pool_size=2)(X)
        #return X
        Z = Flatten()(X)
        
        X1 = Dense(512, activation= 'relu',  activity_regularizer=regularizers.l2(1e-6))(Z)
        X1 = Dense(512, activation= 'relu',  activity_regularizer=regularizers.l2(1e-6))(X1)

        out1 = Dense(nb_classes, activation = 'softmax',name='ct')(X1)
        out2 = Dense(nb_classes, activation = 'softmax',name='cg')(X1)

        if multi_task:
            model = Model(inputs=X_input, outputs = [out1,out2], name='Archi_3DCONV')
        else:
            model = Model(inputs=X_input, outputs = out1, name='Archi_3DCONV')
       

        return model
    
    def Archi_2DCONV(X,nb_classes,multi_task,**conv_params):
        nb_conv = 3
        l2_rate = 1.e-6
        input_shape = X.shape
        X_input = Input(input_shape)
        X = X_input
        
        strides = conv_params.setdefault("strides", 1)
        padding = conv_params.setdefault("padding", "same")
        kernel_regularizer = conv_params.setdefault("kernel_regularizer", regularizers.l2(l2_rate))
        kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")

        s1 = Conv2D(64, 3, activation='relu',input_shape=(64, 64, 5),padding="same",kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(X)
        #s1 = MaxPooling2D(pool_size=2)(s1)
        s1 = BatchNormalization()(s1)
        x1 = s1
        s1 = Conv2D(64, 3, activation='relu', padding="same",kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(s1)
        x2 = s1
        #s1 = MaxPooling2D(pool_size=2)(s1)
        s1 = BatchNormalization()(s1)
    
        merge = concatenate([x1,x2])
        s1 = merge
        s1 = Conv2D(64, 3, activation='relu', padding="same",kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(s1)
        #s1 = MaxPooling2D(pool_size=2)(s1)
        s1 = BatchNormalization()(s1)
        x3 = s1
        merge = concatenate([x2, x3])
        s1 = merge
        s1 = Conv2D(64, 3, activation='relu', padding="same",kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(s1)
        s1 = MaxPooling2D(pool_size=2)(s1)
        s1 = BatchNormalization()(s1)
        X1 = Flatten()(s1)
        X1 = Dense(256, activation= 'relu',  activity_regularizer=regularizers.l2(1e-6))(X1)
        #X1 = Dense(512, activation= 'relu',  activity_regularizer=regularizers.l2(1e-6))(X1)

        out1 = Dense(nb_classes, activation = 'softmax')(X1)
        out2 = Dense(nb_classes, activation = 'softmax')(X1)

        if multi_task:
            model = Model(inputs=X_input, outputs = [out1,out2], name='Archi_2DCONV')
        else:
            model = Model(inputs=X_input, outputs = out1, name='Archi_2DCONV')  
        return model
    def Archi_1DCONV(X,nb_classes,multi_task):
         input_shape = X.shape
         X_input = Input(input_shape)
         l2_rate = 1.e-6
         kernel_regularizer = regularizers.l2(l2_rate)
         kernel_initializer = "he_normal"
         X = X_input
         s1 = Conv1D(64, 5, activation ='relu',kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(X)
         s1 = BatchNormalization()(s1)
         s2 = Conv1D(64, 5, activation = 'relu', padding = 'same',kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(s1)
         s2 = BatchNormalization()(s2)
         s3 = Conv1D(64, 5, activation = 'relu', padding = 'same',kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(s2)
         s3 = BatchNormalization()(s3)
         res_x = tf.keras.layers.add([s1, s3])
         X1 = Flatten()(res_x)
         X1 = Dense(512, activation= 'relu',  activity_regularizer=regularizers.l2(1e-6))(X1)
         X1 = Dense(512, activation= 'relu',  activity_regularizer=regularizers.l2(1e-6))(X1)
         out1 = Dense(3, activation = 'softmax',name="ct",)(X1)
         out2 = Dense(3, activation = 'softmax',name="cg",)(X1)
         if multi_task:
             model = Model(inputs=X_input, outputs = [out1,out2], name='Archi_1DCONV')
         else:
            model = Model(inputs=X_input, outputs = out1, name='Archi_1DCONV')

         return model
    
class TCN:
    def __init__(self,nb_filters,**kwargs):
        self.activation = 'wavenet'
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
        self.activation = 'wavenet'
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
        
        
class Architecture:
    def __init__(self,**kwargs):
        self.model = None
        #self.architecture = kwargs['architecture']
        self.X = kwargs['input']
        self.task = kwargs['multi_task']
        self.nb_classes = 3
        self.input_shape = self.X.shape
        self.X_input = Input(self.input_shape)
        
    def Conv2D_Conv3D_TCN(self):
        
        z = Convolution.Archi_3DCONV(self.X,self.nb_classes, self.task)
        z = tf.keras.layers.Reshape((6*8*5,2))(z)
        X1 = tcn_.TCN(128,dilations = [1, 2, 4, 8, 16, 32], return_sequences=True, activation = 'wavenet',name = 'tnc1')(z)
        X1 = tcn_.TCN(64,dilations = [1, 2, 4, 8, 16, 32], return_sequences=True, activation = 'wavenet',name = 'tnc2')(X1)
        
        X1 = Flatten()(X1)
        X1 = Dense(256, activation= 'relu',  activity_regularizer=regularizers.l2(1e-6))(X1)
        out1 = Dense(3, activation = 'softmax',name="ct",)(X1)
        out2 = Dense(3, activation = 'softmax',name="cg",)(X1)
        inp = Input(self.X.shape)
        model = Model(inputs=self.X_input, outputs = [out1,out2], name='Archi_3D_TCN')

        return model
        """
        z = Convolution.Archi_2DCONV(self.nb_classes, self.task)
        tcn = TCN(activation = 'wavenet', nb_filters=128)
        X2 = tcn.tcn_model(z,nb_filters=128,activation='wavenet')
        X2 = tcn.tcn_model(X2,nb_filters=64,activation='wavenet')
        
        X = concatenate([X1,X2])
        """
        X = Flatten()(X1)
        y1 = Dense(256, activation= 'relu',  activity_regularizer=regularizers.l2(1e-6))(X)
        out1 = Dense(3, activation = 'softmax',name="ct",)(y1)
        out2 = Dense(3, activation = 'softmax',name="cg",)(y1)
        if self.task:
            model = Model(inputs=Input(self.X.shape), outputs = [out1,out2], name='Archi_3D_TCN')
        else:
            model = Model(inputs=Input(X.shape), outputs = out1, name='Archi_3D_TCN') 
        return model
        
    
    def Conv1D_TCN(self,**kwargs):
        X = Convolution.Archi_1DCONV(self.X,self.nb_classes)
        z = Flatten()(X)
        tcn = TCN(activation = 'wavenet',nb_filters=128)
        X = tcn.tcn_model(z,nb_filters=128,activation=kwargs['activation'])
        X = tcn.tcn_model(X,nb_filters=64,activation=kwargs['activation'])
        return X

    def single_task_Conv1D_TCN(self,**kwargs):
        X = self.Conv1D_TCN(**kwargs)
        z = Flatten()(X)

        y1 = Dense(256, activation= 'relu',  activity_regularizer=regularizers.l2(1e-6))(z)
        y1 = Dense(3, activation = 'softmax',name="crop_type",)(y1)
        
        model = Model(inputs=X, outputs = y1, name='temp_cnn')
        return model

        ##TODO: Single task model architecute
        ##TODO: Conv1D, Conv2D, Conv1D_TCN, Conv2D_TCN, TCN -> Single Task Crop Type
        ##TODO: Multi task model architecture
        ##TODO: Conv1D, Conv2D, Conv1D_TCN, Conv2D_TCN, TCN -> Multi Task
       
    def tcn(self,**kwargs):
        input_shape = self.X.shape
        X_input = Input(input_shape)
        X = X_input
        tcn = TCN(activation = 'wavenet',nb_filters=128)
        X = tcn.tcn_model(X,nb_filters=128,activation='wavenet')
        X = tcn.tcn_model(X,nb_filters=64,activation='wavenet')
        z = Flatten()(X)
        y1 = Dense(256, activation= 'relu',  activity_regularizer=regularizers.l2(1e-6))(z)
        out1 = Dense(3, activation = 'softmax',name="ct",)(y1)
        out2 = Dense(3, activation = 'softmax',name="cg",)(y1)
    
        if self.multi_task:
            model = Model(inputs=X_input, outputs = [out1,out2], name='Archi_1DCONV')
        else:
            model = Model(inputs=X_input, outputs = out1, name='Archi_1DCONV')

        return model

        
        
    
def runArchi(noarchi, *args):
	#---- variables
	n_epochs = 20
	batch_size = 32
	
	switcher = {		
		0: Convolution.Archi_3DCONV,
	}
	func = switcher.get(noarchi, lambda: 0)
	model = func(args[0], args[1].shape[1])
    
	
	#if len(args)==5:
	#	return trainTestModel_EarlyAbandon(model, *args, n_epochs=n_epochs, batch_size=batch_size)
	#elif len(args)==7:
	#	return trainValTestModel_EarlyAbandon(model, *args, n_epochs=n_epochs, batch_size=batch_size)
        