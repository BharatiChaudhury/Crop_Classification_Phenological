import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D,Conv2D, BatchNormalization, Flatten, Dense, Input, concatenate,Conv3D,MaxPooling3D,SpatialDropout1D
from keras import regularizers
import keras.backend as K
from keras.layers import Activation, Lambda
import numpy as np


class Convolution:
        
    def Archi_3DCONV(X,nb_classes):
    
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
        X = Conv3D(kernel_size=(3,3,7),filters = 2, activation = 'relu')(X)
        print(X.shape)
        X = MaxPooling3D(pool_size=(2,2,2))(X)
        X = Conv3D(filters=4, kernel_size=(3,3,3), activation='relu')(X)
        X = MaxPooling3D(pool_size=2)(X)
        Z = Flatten()(X)
        y1 = Dense(nbunits_fc,activation='relu',activity_regularizer=regularizers.l2(l2_rate))(Z)
        y2 = Dense(nbunits_fc,activation='relu',activity_regularizer=regularizers.l2(l2_rate))(Z)
        
        out1 = Dense(nb_classes, activation = 'softmax')(y1)
        out2 = Dense(nb_classes, activation = 'softmax')(y1)

        #out = X
        model = Model(inputs=X_input, outputs = [out1,out2], name='Archi_3DCONV')
        return model
    