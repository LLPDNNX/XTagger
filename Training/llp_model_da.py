import keras
import tensorflow as tf
from keras import backend as K
import os
import sys

class Sequence(object):
    def __init__(self,scope=None):
        self._scope = scope
        self._layerList = []
        
    def add(self,layer):
        self._layerList.append(layer)
        
    def __call__(self,x):
        if self._scope!=None:
            with tf.name_scope(self._scope):
                for layer in self._layerList:
                    x = layer(x)
                return x
        for layer in self._layerList:
            x = layer(x)
        return x


class Conv(object):
    def __init__(self,filters,size,stride,activation=keras.layers.LeakyReLU(alpha=0.1),options={}):
        self.conv = keras.layers.Conv1D(
            filters, 
            size, 
            strides=stride, 
            padding='same',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=keras.regularizers.l2(10**(-7)),
        )
        self.dropout = keras.layers.Dropout(0.1)
        self.activation = activation
        
    def __call__(self,x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x

    
class LSTM(object):
    def __init__(self,units,reverse=True,options={}):
        self.lstm = keras.layers.LSTM(units,
            go_backwards=reverse,
            implementation=2,
            recurrent_dropout=0.05, #not possible with CuDNNLSTM
            activation='tanh', #same as CuDNNLSTM
            recurrent_activation='sigmoid',  #same as CuDNNLSTM
            kernel_regularizer=keras.regularizers.l2(10**(-7))
        )
        self.dropout = keras.layers.Dropout(0.1)
        
    def __call__(self,x):
        return self.dropout(self.lstm(x))
    
class Dense(object):
    def __init__(self,nodes,dropout=0.1,activation=keras.layers.LeakyReLU(alpha=0.1),kernel_reg=10**(-7),bias_reg=0,options={}):
        self.dense = keras.layers.Dense(
            nodes,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=keras.regularizers.l2(kernel_reg),
            bias_regularizer=keras.regularizers.l1(bias_reg),
        )
        self.dropout = keras.layers.Dropout(dropout)
        self.activation = activation
    
    def __call__(self,x):
        x = self.dense(x)
        x = self.dropout(x)
        if self.activation!=None:
            x = self.activation(x)
        return x
    
    
class ModelDA(object):
    def __init__(self,nclasses,isParametric=False,useLSTM=True,options={}):
        self.nclasses = nclasses
        self.isParametric = isParametric
        
        self.cpf_conv = Sequence(scope='cpf_conv')
        self.cpf_conv.add(Conv(64,1,1,options=options))
        self.cpf_conv.add(Conv(32,1,1,options=options))
        self.cpf_conv.add(Conv(32,1,1,options=options))
        self.cpf_conv.add(Conv(8,1,1,options=options))
            
        self.npf_conv = Sequence(scope='npf_conv')
        self.npf_conv.add(Conv(32,1,1,options=options))
        self.npf_conv.add(Conv(16,1,1,options=options))
        self.npf_conv.add(Conv(16,1,1,options=options))
        self.npf_conv.add(Conv(4,1,1,options=options))
        
        self.sv_conv = Sequence(scope='sv_conv')
        self.sv_conv.add(Conv(32,1,1,options=options))
        self.sv_conv.add(Conv(16,1,1,options=options))
        self.sv_conv.add(Conv(16,1,1,options=options))
        self.sv_conv.add(Conv(8,1,1,options=options))
            
        if useLSTM:
            self.cpf_lstm = LSTM(150,True,options=options) #8*25=200 inputs
            self.npf_lstm = LSTM(50,True,options=options) #4*25=100 inputs
            self.sv_lstm = LSTM(50,True,options=options) #8*4=32 inputs
        else:
            self.cpf_lstm = keras.layers.Flatten()
            self.npf_lstm = keras.layers.Flatten()
            self.sv_lstm = keras.layers.Flatten()
    
        self.full_features = Sequence(scope='features')
        self.full_features.add(keras.layers.Concatenate())
        self.full_features.add(Dense(200,activation=keras.layers.Activation('tanh',name="features"),options=options))
        #self.full_features.add(keras.layers.GaussianNoise(0.1))
        '''
        self.conv_class_prediction = Sequence(scope='class_prediction')
        self.conv_class_prediction.add(keras.layers.Flatten())
        self.conv_class_prediction.add(keras.layers.Concatenate())
        self.conv_class_prediction.add(Dense(20,options=options))
        self.conv_class_prediction.add(Dense(20,options=options))
        self.conv_class_prediction.add(Dense(nclasses,activation=keras.layers.Softmax(),options=options))
        
        self.lstm_class_prediction = Sequence(scope='class_prediction')
        self.lstm_class_prediction.add(keras.layers.Concatenate())
        self.lstm_class_prediction.add(Dense(20,options=options))
        self.lstm_class_prediction.add(Dense(20,options=options))
        self.lstm_class_prediction.add(Dense(nclasses,activation=keras.layers.Softmax(),options=options))
        '''
        self.full_class_prediction = Sequence(scope='class_prediction')
        self.full_class_prediction.add(Dense(100,options=options))
        self.full_class_prediction.add(Dense(100,options=options))
        self.full_class_prediction.add(Dense(100,options=options))
        self.full_class_prediction.add(Dense(nclasses,activation=keras.layers.Softmax(name="prediction"),options=options))
            
        def gradientReverse(x):
            #backward = tf.negative(x)
            backward = tf.negative(x*tf.exp(tf.abs(x)))
            #backward = tf.negative(x*tf.exp(tf.square(x)))
            forward = tf.identity(x)
            return backward + tf.stop_gradient(forward - backward)

        self.domain_prediction = Sequence(scope='domain_prediction')
        self.domain_prediction.add(keras.layers.Lambda(gradientReverse))
        '''
        self.domain_prediction.add(Dense(100,kernel_reg=0.1,bias_reg=0.01,options=options))
        self.domain_prediction.add(Dense(100,kernel_reg=0.1,bias_reg=0.01,options=options))
        self.domain_prediction.add(Dense(1,kernel_reg=0.1,bias_reg=0.01,activation=None,options=options))
        '''
        self.domain_prediction.add(Dense(100,options=options))
        self.domain_prediction.add(Dense(100,options=options))
        self.domain_prediction.add(Dense(1,activation=keras.layers.Activation('sigmoid'),options=options))
            
    def extractFeatures(self,globalvars,cpf,npf,sv,gen=None):
        cpf_conv = self.cpf_conv(cpf)
        npf_conv = self.npf_conv(npf)
        sv_conv = self.sv_conv(sv)
        
        cpf_lstm = self.cpf_lstm(cpf_conv)
        npf_lstm = self.npf_lstm(npf_conv)
        sv_lstm = self.sv_lstm(sv_conv)
        
        if self.isParametric:
            full_features = self.full_features([globalvars,gen,cpf_lstm,npf_lstm,sv_lstm])
        else:
            full_features = self.full_features([globalvars,cpf_lstm,npf_lstm,sv_lstm])
            
        return full_features
    
    def predictClass(self,globalvars,cpf,npf,sv,gen=None):
        full_features = self.extractFeatures(globalvars,cpf,npf,sv,gen)
        full_class_prediction = self.full_class_prediction(full_features)
        return full_class_prediction
        
    def predictDomain(self,globalvars,cpf,npf,sv,gen=None):
        full_features = self.extractFeatures(globalvars,cpf,npf,sv,gen)
        
        
        #TODO: add gradient reversal layer here!!!
        domain_prediction = self.domain_prediction(full_features)
        return domain_prediction

    
    
