import keras
import tensorflow as tf
from keras import backend as K
import os
import sys
import logging
import xtools
    
class AttentionNetwork(xtools.NominalNetwork):
    def __init__(self,featureDict):
        xtools.NominalNetwork.__init__(self,featureDict)
        
        
        #### CPF ####
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
            
        self.cpf_attention = []#keras.layers.Lambda(lambda x: tf.transpose(x,[0,2,1]))]
        for i,filters in enumerate([64,32,32,10]):
            self.cpf_attention.extend([
                keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name='cpf_attention'+str(i+1)
                ),
            ])
            if i<3:
                self.cpf_attention.extend([
                    keras.layers.LeakyReLU(alpha=0.1,name='cpf_attention_activation'+str(i+1)),
                    keras.layers.Dropout(0.1,name='cpf_attention_dropout'+str(i+1)),
                ])
            else:
                self.cpf_attention.extend([
                    #keras.layers.Activation('sigmoid',name="cpf_attention_activation"+str(i+1)),
                    keras.layers.Dropout(0.1,name='cpf_attention_dropout'+str(i+1)),
                ])

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

        self.npf_attention = []#keras.layers.Lambda(lambda x: tf.transpose(x,[0,2,1]))]
        for i,filters in enumerate([64,32,32,10]):
            self.npf_attention.extend([keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name="npf_attention"+str(i+1)
                ),
            ])
            if i<3:
                self.npf_attention.extend([
                    keras.layers.LeakyReLU(alpha=0.1,name="npf_attention_activation"+str(i+1)),
                    keras.layers.Dropout(0.1,name="npf_attention_droupout"+str(i+1)),
                ])
            else:
                self.npf_attention.extend([
                    #keras.layers.Activation('sigmoid',name="npf_attention_activation"+str(i+1)),
                    keras.layers.Dropout(0.1,name="npf_attention_droupout"+str(i+1)),
                ])
        
        #### Global ####
        self.global_preproc = \
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["globalvars"]["branches"],
                self.featureDict["globalvars"]["preprocessing"]
            ), name="global_preproc")
        
        
        #### Features ####
        self.full_features = [keras.layers.Concatenate()]
        for i,nodes in enumerate([200]):
            self.full_features.extend([
                keras.layers.Dense(
                    nodes,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name="features_dense"+str(i+1)
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
        
    def applyAttention(self,features,attention):
        result = keras.layers.Lambda(lambda x: tf.matmul(tf.transpose(x[0],[0,2,1]),x[1]))([attention,features])
        return keras.layers.Flatten()(result)
        #result = tf.constant(0,dtype=tf.float32,shape=[tf.getShape(features)[0],attention.shape[1],features.shape[2]
            
 
    def extractFeatures(self,globalvars,cpf,npf,sv,muon,electron,gen=None):
        globalvars_preproc = self.global_preproc(globalvars)
        
        cpf_conv = self.applyLayers(self.cpf_preproc(cpf),self.cpf_conv)
        npf_conv = self.applyLayers(self.npf_preproc(npf),self.npf_conv)
        sv_conv = self.applyLayers(self.sv_preproc(sv),self.sv_conv)
        muon_conv = self.applyLayers(self.muon_preproc(muon),self.muon_conv)
        electron_conv = self.applyLayers(self.electron_preproc(electron),self.electron_conv)
                
        cpf_attention = self.applyLayers(self.cpf_preproc(cpf),self.cpf_attention)
        npf_attention = self.applyLayers(self.npf_preproc(npf),self.npf_attention)
        
        cpf_tensor = self.applyAttention(cpf_conv,cpf_attention)
        npf_tensor = self.applyAttention(npf_conv,npf_attention)
        
        
        
        
        full_features = self.applyLayers([globalvars_preproc,cpf_tensor,npf_tensor,sv_conv,muon_conv,electron_conv,gen], self.full_features)
        #full_features = self.applyLayers([globalvars_preproc,cpf_conv,npf_conv,sv_conv,muon_conv,gen], self.full_features)
        #full_features = self.applyLayers([globalvars_preproc,cpf_conv,npf_conv,sv_conv,gen], self.full_features)
        
        return full_features
    
network = AttentionNetwork
    
