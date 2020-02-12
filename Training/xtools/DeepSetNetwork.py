import keras
import tensorflow as tf
from keras import backend as K
import os
import sys
import logging
from NominalNetwork import NominalNetwork
    
class DeepSetNetwork(NominalNetwork):
    def __init__(self,featureDict):
        NominalNetwork.__init__(self,featureDict)
        
        self.cpf_conv = []
        for i,filters in enumerate([64,32,32,8]):
            self.cpf_conv.extend([
                keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name='cpf_conv'+str(i+1)
                ),
                keras.layers.LeakyReLU(alpha=0.1,name='cpf_activation'+str(i+1)),
                keras.layers.Dropout(0.1,name='cpf_dropout'+str(i+1)),
            ])
        self.cpf_conv.append(keras.layers.Lambda(lambda x: tf.reduce_sum(x,axis=1),name='cpf_sum'))
        

        #### NPF ####
        self.npf_conv = []
        for i,filters in enumerate([32,16,16,4]):
            self.npf_conv.extend([keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name="npf_conv"+str(i+1)
                ),
                keras.layers.LeakyReLU(alpha=0.1,name="npf_activation"+str(i+1)),
                keras.layers.Dropout(0.1,name="npf_droupout"+str(i+1)),
            ])
        self.npf_conv.append(keras.layers.Lambda(lambda x: tf.reduce_sum(x,axis=1),name="npf_sum"))
        
        
        #### SV ####
        self.sv_conv = []
        for i,filters in enumerate([32,16,16,8]):
            self.sv_conv.extend([keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name="sv_conv"+str(i+1)
                ),
                keras.layers.LeakyReLU(alpha=0.1,name="sv_activation"+str(i+1)),
                keras.layers.Dropout(0.1,name="sv_dropout"+str(i+1)),
            ])
        self.sv_conv.append(keras.layers.Lambda(lambda x: tf.reduce_sum(x,axis=1),name="sv_sum"))
        
        
        #### Muons ####
        self.muon_conv = []
        for i,filters in enumerate([32,16,16,12]):
            self.muon_conv.extend([keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name="muon_conv"+str(i+1)
                ),
                keras.layers.LeakyReLU(alpha=0.1,name="muon_activation"+str(i+1)),
                keras.layers.Dropout(0.1,name="muon_dropout"+str(i+1)),
            ])
        self.muon_conv.append(keras.layers.Lambda(lambda x: tf.reduce_sum(x,axis=1),name="muon_sum"))
        
        
        #### Electron ####
        self.electron_conv = []
        for i,filters in enumerate([32,16,16,12]):
            self.electron_conv.extend([keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name="electron_conv"+str(i+1)
                ),
                keras.layers.LeakyReLU(alpha=0.1,name="electron_activation"+str(i+1)),
                keras.layers.Dropout(0.1,name="electron_dropout"+str(i+1)),
            ])
        self.electron_conv.append(keras.layers.Lambda(lambda x: tf.reduce_sum(x,axis=1),name="electron_sum"))
        
        
        #### Features ####
        self.full_features = [keras.layers.Concatenate()]
        for i,nodes in enumerate([200]):
            self.full_features.extend([
                keras.layers.Dense(
                    200,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name="features_dense"+str(i)
                ),
                keras.layers.LeakyReLU(alpha=0.1,name="features_activation"+str(i+1))
            ])
            
            
        #### Class prediction ####
        self.class_prediction = []
        for i,nodes in enumerate([100,100]):
            self.class_prediction.extend([
                keras.layers.Dense(
                    nodes,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name="class_dense"+str(i+1)
                ),
                keras.layers.LeakyReLU(alpha=0.1,name="class_activation"+str(i+1)),
                keras.layers.Dropout(0.1,name="class_dropout"+str(i+1)),
            ])
        self.class_prediction.extend([
            keras.layers.Dense(
                self.nclasses,
                kernel_initializer='lecun_normal',
                bias_initializer='zeros',
                kernel_regularizer=keras.regularizers.l1(1e-6),
                name="class_nclasses"
            ),
            keras.layers.Softmax(name="class_softmax")
        ])

    
 
    def extractFeatures(self,globalvars,cpf,npf,sv,muon,electron,gen=None):
        globalvars_preproc = self.global_preproc(globalvars)
        
        cpf_conv = self.applyLayers(self.cpf_preproc(cpf),self.cpf_conv)
        npf_conv = self.applyLayers(self.npf_preproc(npf),self.npf_conv)
        sv_conv = self.applyLayers(self.sv_preproc(sv),self.sv_conv)
        muon_conv = self.applyLayers(self.muon_preproc(muon),self.muon_conv)
        electron_conv = self.applyLayers(self.electron_preproc(electron),self.electron_conv)
        
        full_features = self.applyLayers([globalvars_preproc,cpf_conv,npf_conv,sv_conv,muon_conv,electron_conv,gen], self.full_features)
        #full_features = self.applyLayers([globalvars_preproc,cpf_conv,npf_conv,sv_conv,muon_conv,gen], self.full_features)
        #full_features = self.applyLayers([globalvars_preproc,cpf_conv,npf_conv,sv_conv,gen], self.full_features)
        
        return full_features
    

    
