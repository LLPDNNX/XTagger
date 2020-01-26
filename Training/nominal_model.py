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
            with tf.variable_scope(self._scope):
                for layer in self._layerList:
                    x = layer(x)
                return x
        for layer in self._layerList:
            x = layer(x)
        return x


class Conv(object):
    def __init__(self,filters,size,stride,activation=keras.layers.LeakyReLU(alpha=0.1),options={},name=None):
        self.conv = keras.layers.Conv1D(
            filters, 
            size, 
            strides=stride, 
            padding='same',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=keras.regularizers.l2(10**(-7)),
            name=name
        )
        self.dropout = keras.layers.Dropout(0.1)
        self.activation = activation
        
    def __call__(self,x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x

    
class LSTM(object):
    def __init__(self,units,reverse=True,options={},name=None):
        self.lstm = keras.layers.LSTM(units,
            go_backwards=reverse,
            implementation=2,
            recurrent_dropout=0.05, #not possible with CuDNNLSTM
            activation='tanh', #same as CuDNNLSTM
            recurrent_activation='sigmoid',  #same as CuDNNLSTM
            kernel_regularizer=keras.regularizers.l2(10**(-7)),
            name=name
        )
        self.dropout = keras.layers.Dropout(0.1)
        
    def __call__(self,x):
        return self.dropout(self.lstm(x))
    
class Dense(object):
    def __init__(self,nodes,dropout=0.1,activation=keras.layers.LeakyReLU(alpha=0.1),kernel_reg=10**(-7),bias_reg=0,kernel_constraint=None,bias_constraint=None,options={},name=None):
        self.dense = keras.layers.Dense(
            nodes,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=keras.regularizers.l2(kernel_reg),
            bias_regularizer=keras.regularizers.l1(bias_reg),
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=name
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
    def __init__(self,featureDict,isParametric=False,useLSTM=True,useWasserstein=False,options={}):
        self.featureDict = featureDict
        self.nclasses = len(self.featureDict["truth"]["branches"])
        self.isParametric = isParametric
        self.useWasserstein = useWasserstein
        
        with tf.variable_scope("cpf_conv"):
            self.cpf_preproc = keras.layers.Lambda(
                self.preprocessingFct(self.featureDict["cpf"]["branches"],self.featureDict["cpf"]["preprocessing"])
            )
        
            self.cpf_conv = Sequence(scope='cpf_conv')
            self.cpf_conv.add(self.cpf_preproc)
            #self.cpf_conv.add(keras.layers.BatchNormalization())
            self.cpf_conv.add(Conv(64,1,1,options=options,name="cpf_conv1"))
            self.cpf_conv.add(Conv(32,1,1,options=options,name="cpf_conv2"))
            self.cpf_conv.add(Conv(32,1,1,options=options,name="cpf_conv3"))
            self.cpf_conv.add(Conv(8,1,1,options=options,name="cpf_conv4"))
            
        with tf.variable_scope("npf_conv"):
            self.npf_preproc = keras.layers.Lambda(
                self.preprocessingFct(self.featureDict["npf"]["branches"],self.featureDict["npf"]["preprocessing"])
            )
            self.npf_conv = Sequence(scope='npf_conv')
            self.npf_conv.add(self.npf_preproc)
            #self.npf_conv.add(keras.layers.BatchNormalization())
            self.npf_conv.add(Conv(32,1,1,options=options,name="npf_conv1"))
            self.npf_conv.add(Conv(16,1,1,options=options,name="npf_conv2"))
            self.npf_conv.add(Conv(16,1,1,options=options,name="npf_conv3"))
            self.npf_conv.add(Conv(4,1,1,options=options,name="npf_conv4"))
        
        with tf.variable_scope("sv_conv"):
            self.sv_preproc = keras.layers.Lambda(
                self.preprocessingFct(self.featureDict["sv"]["branches"],self.featureDict["sv"]["preprocessing"])
            )
        
            self.sv_conv = Sequence(scope='sv_conv')
            self.sv_conv.add(self.sv_preproc)
            #self.sv_conv.add(keras.layers.BatchNormalization())
            self.sv_conv.add(Conv(32,1,1,options=options,name="sv_conv1"))
            self.sv_conv.add(Conv(16,1,1,options=options,name="sv_conv2"))
            self.sv_conv.add(Conv(16,1,1,options=options,name="sv_conv3"))
            self.sv_conv.add(Conv(8,1,1,options=options,name="sv_conv4"))
            
        with tf.variable_scope("lstm"):
            if useLSTM:
                self.cpf_lstm = LSTM(150,True,options=options,name="cpf_lstm1") #8*25=200 inputs
                self.npf_lstm = LSTM(50,True,options=options,name="npf_lstm1") #4*25=100 inputs
                self.sv_lstm = LSTM(50,True,options=options,name="sv_lstm1") #8*4=32 inputs
            else:
                self.cpf_lstm = keras.layers.Flatten()
                self.npf_lstm = keras.layers.Flatten()
                self.sv_lstm = keras.layers.Flatten()
    
        with tf.variable_scope("features"):
            self.global_preproc = keras.layers.Lambda(
                self.preprocessingFct(self.featureDict["globalvars"]["branches"],self.featureDict["globalvars"]["preprocessing"])
            )
            self.full_features = Sequence(scope='features')
            self.full_features.add(keras.layers.Concatenate())
            self.full_features.add(Dense(200,options=options,name="features1",activation=None))
            self.full_features.add(keras.layers.Activation('tanh',name="features2"))
            #self.full_features.add(keras.layers.GaussianNoise(0.1))

        with tf.variable_scope("class_prediction"):
            self.full_class_prediction = Sequence(scope='class_prediction')
            self.full_class_prediction.add(Dense(100,options=options))
            self.full_class_prediction.add(Dense(100,options=options))
            self.full_class_prediction.add(Dense(self.nclasses,dropout=0,activation=keras.layers.Softmax(name="prediction"),options=options))

        with tf.variable_scope("domain_prediction"):
            def gradientReverse(x):
                if useWasserstein:
                    backward = tf.negative(x)
                else:
                    backward = tf.negative(x)
                forward = tf.identity(x)
                return (backward + tf.stop_gradient(forward - backward))
                
            self.domain_prediction = Sequence(scope='domain_prediction')
            self.domain_prediction.add(keras.layers.Lambda(gradientReverse))
            
            if useWasserstein:
                self.domain_prediction.add(Dense(50,dropout=0.1,kernel_reg=0.01,bias_reg=0.01,kernel_constraint=keras.constraints.max_norm(0.01),bias_constraint=keras.constraints.max_norm(0.01),options=options))
                self.domain_prediction.add(Dense(50,dropout=0.1,kernel_reg=0.01,bias_reg=0.01,kernel_constraint=keras.constraints.max_norm(0.01),bias_constraint=keras.constraints.max_norm(0.01),options=options))
                self.domain_prediction.add(Dense(1,dropout=0,kernel_reg=0.01,bias_reg=0.01,kernel_constraint=keras.constraints.max_norm(0.01),bias_constraint=keras.constraints.max_norm(0.01),activation=keras.layers.Activation('relu'),options=options))
            else:
                self.domain_prediction.add(Dense(50,options=options,dropout=0.1))
                self.domain_prediction.add(Dense(50,options=options,dropout=0.1))
                self.domain_prediction.add(Dense(1,activation=keras.layers.Activation('sigmoid'),options=options,dropout=0))

    def preprocessingFct(self,featureNames,preprocDict):
        def applyPreproc(inputFeatures):
            unstackFeatures = tf.unstack(inputFeatures,axis=-1)
            if len(unstackFeatures)!=len(featureNames):
                print "ERROR: Number of features (",len(unstackFeatures),") does not match given list of names (",len(featureNames),"): ",featureNames
                sys.exit(1)
            unusedPreproc = preprocDict.keys()
            if len(unusedPreproc)==0:
                return inputFeatures
            for i,featureName in enumerate(featureNames):
                if featureName in unusedPreproc:
                    unusedPreproc.remove(featureName)
                if featureName in preprocDict.keys():
                    unstackFeatures[i] = preprocDict[featureName](unstackFeatures[i])
                   
            if len(unusedPreproc)>0:
                print "WARNING - following preprocessing not applied: ",unusedPreproc
            return tf.stack(unstackFeatures,axis=-1)
        return applyPreproc
        
    def getPreprocFeatures(self,globalvars,cpf,npf,sv):
        return [self.global_preproc(globalvars),self.cpf_preproc(cpf),self.npf_preproc(npf),self.sv_preproc(sv)]
    
            
    def extractFeatures(self,globalvars,cpf,npf,sv,gen=None):
        cpf_conv = self.cpf_conv(cpf)
        npf_conv = self.npf_conv(npf)
        sv_conv = self.sv_conv(sv)
        
        cpf_lstm = self.cpf_lstm(cpf_conv)
        npf_lstm = self.npf_lstm(npf_conv)
        sv_lstm = self.sv_lstm(sv_conv)
        
        globalvars_preproc = self.global_preproc(globalvars)
        
        if self.isParametric:
            full_features = self.full_features([gen,globalvars_preproc,cpf_lstm,npf_lstm,sv_lstm])
        else:
            full_features = self.full_features([globalvars_preproc,cpf_lstm,npf_lstm,sv_lstm])
            
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

    
    
