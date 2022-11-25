import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D,Conv2D,MaxPooling2D, BatchNormalization, MaxPool2D, Flatten, Dense, Input, concatenate,Conv3D,MaxPooling3D,SpatialDropout1D
from keras import regularizers
import keras.backend as K
from keras.layers import Activation, Lambda

def channel_normalization(x):
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out


def wave_net_activation(x):
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)
    return tf. keras.layers.multiply([tanh_out, sigm_out])


def residual_block(x, s, i, activation, nb_filters, kernel_size, padding, dropout_rate=0, name=''):


    original_x = x
    conv = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding=padding,
                  name=name + '_dilated_conv_%d_tanh_s%d' % (i, s))(x)
    if activation == 'norm_relu':
        x = Activation('relu')(conv)
        x = Lambda(channel_normalization)(x)
    elif activation == 'wavenet':
        x = wave_net_activation(conv)
    else:
        x = Activation(activation)(conv)

    x = SpatialDropout1D(dropout_rate, name=name + '_spatial_dropout1d_%d_s%d_%f' % (i, s, dropout_rate))(x)

    # 1x1 conv.
    x = Conv1D(nb_filters, 1, padding='same')(x)
    res_x = tf.keras.layers.add([original_x, x])
    return res_x, x


def process_dilations(dilations):
    def is_power_of_two(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if all([is_power_of_two(i) for i in dilations]):
        return dilations

    else:
        new_dilations = [2 ** i for i in dilations]
        # print(f'Updated dilations from {dilations} to {new_dilations} because of backwards compatibility.')
        return new_dilations


class TCN:

    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=None,
                 activation='norm_relu',
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=True,
                 name='tcn'):
        self.name = name
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.activation = activation
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding


    def __call__(self, inputs):
        if self.dilations is None:
            self.dilations = [1, 2, 4, 8, 16, 32]
        x = inputs
        x = Conv1D(self.nb_filters, 1, padding=self.padding, name=self.name + '_initial_conv')(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for i in self.dilations:
                x, skip_out = residual_block(x, s, i, self.activation, self.nb_filters,
                                             self.kernel_size, self.padding, self.dropout_rate, name=self.name)
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            x = tf.keras.layers.add(skip_connections)
        x = Activation('relu')(x)

        if not self.return_sequences:
            output_slice_index = -1
            x = Lambda(lambda tt: tt[:, output_slice_index, :])(x)
        return x
def Conv1D_Conv2D_TCN(input_shape1, input_shape2, multitask):
    input1 = Input(input_shape1)
    input2 = Input(input_shape2)
    
    s1 = Conv1D(64, 5, activation ='relu', input_shape = input_shape1[1:])(input1)
    s2 = Conv1D(64, 5, activation = 'relu', padding = 'same')(s1)
    s3 = Conv1D(64, 5, activation = 'relu', padding = 'same')(s2)
    
    
    ## Temporal Convolution Modeling
    x1 = TCN(128,dilations = [1, 2, 4, 8, 16, 32], return_sequences=True, activation = 'wavenet',name = 'tnc1')(input1)
    x2 = TCN(64,dilations = [1, 2, 4, 8, 16, 32], return_sequences=True, activation = 'wavenet',name = 'tnc2')(x1)
    
    
    s4 = Conv1D(64, 30, activation ='relu', input_shape = input_shape1[1:])(input1)
    s5 = Conv1D(64, 30, activation = 'relu', padding = 'same')(s4)
    s6 = Conv1D(64, 30, activation = 'relu', padding = 'same')(s5)
    res_x = tf.keras.layers.add([s4, s6])
    X1 = Flatten()(res_x)
    
    X1 = TCN(128,dilations = [1, 2, 4, 8, 16, 32], return_sequences=True, activation = 'wavenet',name = 'tnc3')(X1)
    X1 = TCN(64,dilations = [1, 2, 4, 8, 16, 32], return_sequences=True, activation = 'wavenet',name = 'tnc4')(X1)
    
    x4 = Flatten()(X1)
    #fused = concatenate([x2,x4])
    #z = Flatten()(fused)
    z = x4
    y1 = Dense(256, activation= 'relu',  activity_regularizer=regularizers.l2(0.001))(z)
    y1 = Dense(3, activation = 'softmax',name="cg",)(y1)
    
    y2 = Dense(256,activation = 'relu', activity_regularizer=regularizers.l2(0.001))(z)
    y2 = Dense(3, activation = 'softmax',name="ct")(y2)
    if multitask:
        model = Model(inputs=[input1,input2], outputs=[y1,y2], name="temp_cnn")
    else:
        model = Model(inputs=[input1,input2], outputs = y1, name='temp_cnn')
    return model
def Conv2D_3D_TCN(input_shape1,input_shape2,multitask,fusion):
    input1 = Input(input_shape1)
    input2 = Input(input_shape2)
    l2_rate = 1.e-6
    kernel_regularizer = regularizers.l2(l2_rate)
    kernel_initializer = "he_normal"
    
    
    X = Conv3D(kernel_size=(3,3,7),filters = 4, activation = 'relu',kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(input1)
    X = MaxPooling3D(pool_size=(2,2,2))(X)
    X = BatchNormalization()(X)
    X = Conv3D(filters=2, kernel_size=(3,3,3), activation='relu',kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(X)
    X = MaxPooling3D(pool_size=2)(X)
    X = BatchNormalization()(X)

    z = Flatten()(X)    
    z = tf.keras.layers.Reshape((3*8*5,4))(X)
    z = Flatten()(z)
    
    
    
    s1 = Conv2D(64, 3, activation='relu',input_shape=(64, 64, 5),padding="same",kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(input2)
    s1 = BatchNormalization()(s1)
    x1 = s1
    s1 = Conv2D(64, 3, activation='relu', padding="same",kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(s1)
    x2 = s1


    s1 = BatchNormalization()(s1)

    merge = concatenate([x1,x2])
    s1 = merge
    s1 = Conv2D(64, 3, activation='relu', padding="same",kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(s1)
    s1 = MaxPooling2D(pool_size=2)(s1)
    s1 = BatchNormalization()(s1)
    x3 = s1
    merge = concatenate([x2, x3])
    s1 = merge
    s1 = Conv2D(64, 3, activation='relu', padding="same",kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(s1)
    s1 = BatchNormalization()(s1)
    z1 = tf.keras.layers.Reshape((32,2*64*64))(s1)
    z1 = Flatten()(z1)
    X1 = concatenate([z,z1])

    X1 = tf.keras.layers.Reshape((32,8207))(X1)
    
    z1 = X1
    
    X2 = TCN(128,dilations = [1, 2, 4, 8], return_sequences=True, activation = 'wavenet', name = 'tnc3')(z1)
    X2 = TCN(128,dilations = [1, 2, 4, 8], return_sequences=True, activation = 'wavenet', name = 'tnc4')(X2)
            
    X2 = Flatten()(X2)
    
    X1 = X2
    
    X1 = Dense(512, activation= 'relu',  activity_regularizer=regularizers.l2(l2_rate))(X1)
    X1 = Dense(512, activation= 'relu',  activity_regularizer=regularizers.l2(l2_rate))(X1)
    
    out1 = Dense(3, activation = 'softmax',name="ct",)(X1)
    out2 = Dense(3, activation = 'softmax',name="cg",)(X1)
    if fusion:
        if multitask:
            model = Model(inputs=[input1,input2], outputs = [out1,out2], name='Archi_3D_TCN')
        else:
            model = Model(inputs=[input1,input2], outputs = out1, name='Archi_3D_TCN')
    else:
        if multitask:
            model = Model(inputs=input1, outputs = [out1,out2], name='Archi_3D_TCN')
        else:
            model = Model(inputs=input1, outputs = out1, name='Archi_3D_TCN')
    return model
    
def mx_tcn_model(input_shape1, multitask):
    input1 = Input(input_shape1)
    
    ## Temporal Convolution Modeling
    x1 = TCN(128,dilations = [1, 2, 4, 8, 16, 32], return_sequences=True, activation = 'wavenet',name = 'tnc1')(input1)
    x2 = TCN(64,dilations = [1, 2, 4, 8, 16, 32], return_sequences=True, activation = 'wavenet',name = 'tnc2')(x1)
    
    z = Flatten()(x2)
    dropout = Dropout(0.5)
    z = dropout(z,training=True)
    y1 = Dense(256, activation= 'relu',  activity_regularizer=regularizers.l2(0.001))(z)
    y1 = Dense(3, activation = 'softmax',name="cg",)(y1)
    
    y2 = Dense(256,activation = 'relu', activity_regularizer=regularizers.l2(0.001))(z)
    y2 = Dense(3, activation = 'softmax',name="ct")(y2)
    if multitask:
        model = Model(inputs=input1, outputs=[y1,y2], name="temp_cnn")
    else:
        model = Model(inputs=input1, outputs = y1, name='temp_cnn')
    return model
#model = model((5,4096),(30,1287),1)
#print(model.summary())
