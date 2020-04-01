import keras
import tensorflow as tf
from keras import backend as K
import os
import sys
import logging
    
class NominalNetwork():
    def __init__(self,featureDict):
        self.featureDict = featureDict
        self.nclasses = len(self.featureDict["truth"]["branches"])
        
        #### inputs #####
        
        self.input_gen = keras.layers.Input(
            shape=(len(self.featureDict["gen"]["branches"]),),
            name="input_gen"
        )
        self.input_globalvars = keras.layers.Input(
            shape=(len(self.featureDict["globalvars"]["branches"]),),
            name="input_global"
        )
        self.input_cpf = keras.layers.Input(
            shape=(self.featureDict["cpf"]["max"], len(self.featureDict["cpf"]["branches"])),
            name="input_cpf"
        )
        self.input_npf = keras.layers.Input(
            shape=(self.featureDict["npf"]["max"], len(self.featureDict["npf"]["branches"])),
            name="input_npf"
        )
        self.input_sv = keras.layers.Input(
            shape=(self.featureDict["sv"]["max"], len(self.featureDict["sv"]["branches"])),
            name="input_sv"
        )
        self.input_muon = keras.layers.Input(
            shape=(self.featureDict["muon"]["max"], len(self.featureDict["muon"]["branches"])),
            name="input_muon"
        )
        self.input_electron = keras.layers.Input(
            shape=(self.featureDict["electron"]["max"], len(self.featureDict["electron"]["branches"])),
            name="input_electron"
        )
        
        
        #### CPF ####
        self.cpf_preproc = \
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["cpf"]["branches"],
                self.featureDict["cpf"]["preprocessing"]
            ),name='cpf_preproc')
        
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
        self.cpf_conv.append(keras.layers.Flatten(name='cpf_flatten'))
        

        #### NPF ####
        self.npf_preproc = \
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["npf"]["branches"],
                self.featureDict["npf"]["preprocessing"]
            ),name='npf_preproc')
        
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
        self.npf_conv.append(keras.layers.Flatten(name="npf_flatten"))
        
        
        #### SV ####
        self.sv_preproc = \
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["sv"]["branches"],
                self.featureDict["sv"]["preprocessing"]
            ),name='sv_preproc')
        
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
        self.sv_conv.append(keras.layers.Flatten(name="sv_flatten"))
        
        
        #### Muons ####
        self.muon_preproc = \
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["muon"]["branches"],
                self.featureDict["muon"]["preprocessing"]
            ),name="muon_preproc")
        
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
        self.muon_conv.append(keras.layers.Flatten(name="muon_flatten"))
        
        
        #### Electron ####
        self.electron_preproc = \
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["electron"]["branches"],
                self.featureDict["electron"]["preprocessing"]
            ),name="electron_preproc")
        
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
        self.electron_conv.append(keras.layers.Flatten(name="electron_flatten"))
        
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

    def preprocessingFct(self,featureNames,preprocDict):
        def applyPreproc(inputFeatures):
            unstackFeatures = tf.unstack(inputFeatures,axis=-1)
            if len(unstackFeatures)!=len(featureNames):
                logging.critical("Number of features ("+str(len(unstackFeatures))+") does not match given list of names ("+str(len(featureNames))+"): "+str(featureNames))
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
                logging.warning("Following preprocessing not applied: "+str(unusedPreproc))
            return tf.stack(unstackFeatures,axis=-1)
        return applyPreproc
        
        
    def applyLayers(self,inputTensor,layerList):
        output = layerList[0](inputTensor)
        for layer in layerList[1:]:
            output = layer(output)
        return output
        
    def makePreprocessingModel(self):

        model = keras.models.Model(
            inputs=[
                self.input_gen,
                self.input_globalvars,
                self.input_cpf,
                self.input_npf,
                self.input_sv,
                self.input_muon,
                self.input_electron
            ],
            outputs=[
                gen,
                self.global_preproc(self.input_globalvars),
                self.cpf_preprocself.input_(cpf),
                self.npf_preproc(self.input_npf),
                self.sv_preproc(self.input_sv),
                self.muon_preproc(self.input_muon),
                self.electron_preproc(self.input_electron)
            ])
        return model
        
 
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
    
    def predictClass(self,globalvars,cpf,npf,sv,muon,electron,gen):
        full_features = self.extractFeatures(globalvars,cpf,npf,sv,muon,electron,gen)
        class_prediction = self.applyLayers(full_features,self.class_prediction)
        return class_prediction
        
    def makeClassModel(self):         
        predictedClass = self.predictClass(
            self.input_globalvars,
            self.input_cpf,
            self.input_npf,
            self.input_sv,
            self.input_muon,
            self.input_electron,
            self.input_gen
        )
        model = keras.models.Model(
            inputs=[
                self.input_gen,
                self.input_globalvars,
                self.input_cpf,
                self.input_npf,
                self.input_sv,
                self.input_muon,
                self.input_electron
            ],
            outputs=[
                predictedClass
            ]
        )
        return model
        
network = NominalNetwork
