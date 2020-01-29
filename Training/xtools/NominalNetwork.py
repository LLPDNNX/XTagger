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
        
        #### CPF ####
        self.cpf_conv = [
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["cpf"]["branches"],
                self.featureDict["cpf"]["preprocessing"]
            ))
        ]
        for filters in [64,32,32,8]:
            self.cpf_conv.extend([
                keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6)
                ),
                keras.layers.LeakyReLU(alpha=0.1),
                keras.layers.Dropout(0.1),
            ])
        self.cpf_conv.append(keras.layers.Flatten())
        

        #### NPF ####
        self.npf_conv = [
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["npf"]["branches"],
                self.featureDict["npf"]["preprocessing"]
            ))
        ]
        for filters in [32,16,16,4]:
            self.npf_conv.extend([keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6)
                ),
                keras.layers.LeakyReLU(alpha=0.1),
                keras.layers.Dropout(0.1),
            ])
        self.npf_conv.append(keras.layers.Flatten())
        
        
        #### SV ####
        self.sv_conv = [
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["sv"]["branches"],
                self.featureDict["sv"]["preprocessing"]
            ))
        ]
        for filters in [32,16,16,8]:
            self.sv_conv.extend([keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6)
                ),
                keras.layers.LeakyReLU(alpha=0.1),
                keras.layers.Dropout(0.1),
            ])
        self.sv_conv.append(keras.layers.Flatten())
        
        
        #### Muons ####
        self.muon_conv = [
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["muon"]["branches"],
                self.featureDict["muon"]["preprocessing"]
            ))
        ]
        for filters in [32,16,16,8]:
            self.muon_conv.extend([keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6)
                ),
                keras.layers.LeakyReLU(alpha=0.1),
                keras.layers.Dropout(0.1),
            ])
        self.muon_conv.append(keras.layers.Flatten())
        
        '''
        #### Electron ####
        self.electron_conv = [
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["electron"]["branches"],
                self.featureDict["electron"]["preprocessing"]
            ))
        ]
        for filters in [32,16,16,8]:
            self.electron_conv.extend([keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6)
                ),
                keras.layers.LeakyReLU(alpha=0.1),
                keras.layers.Dropout(0.1),
            ])
        self.electron_conv.append(keras.layers.Flatten())
        '''
        
        #### Global ####
        self.global_preproc = keras.layers.Lambda(
            self.preprocessingFct(self.featureDict["globalvars"]["branches"],self.featureDict["globalvars"]["preprocessing"])
        )
        
        
        #### Features ####
        self.full_features = [
            keras.layers.Concatenate(),
            keras.layers.Dense(
                200,
                kernel_initializer='lecun_normal',
                bias_initializer='zeros',
                kernel_regularizer=keras.regularizers.l1(1e-6)
            ),
            keras.layers.Activation('tanh'),
        ]
            
            
        #### Class prediction ####
        self.class_prediction = []
        for nodes in [100,100]:
            self.class_prediction.extend([
                keras.layers.Dense(
                    200,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6)
                ),
                keras.layers.LeakyReLU(alpha=0.1),
                keras.layers.Dropout(0.1),
            ])
        self.class_prediction.extend([
            keras.layers.Dense(
                self.nclasses,
                kernel_initializer='lecun_normal',
                bias_initializer='zeros',
                kernel_regularizer=keras.regularizers.l1(1e-6)
            ),
            keras.layers.Softmax(name="prediction")
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
 
    def extractFeatures(self,globalvars,cpf,npf,sv,muon,gen=None):
        globalvars_preproc = self.global_preproc(globalvars)
        
        cpf_conv = self.applyLayers(cpf,self.cpf_conv)
        npf_conv = self.applyLayers(npf,self.npf_conv)
        sv_conv = self.applyLayers(sv,self.sv_conv)
        muon_conv = self.applyLayers(muon,self.muon_conv)
        #electron_conv = self.applyLayers(electron,self.electron_conv)
        
        full_features = self.applyLayers([globalvars_preproc,cpf_conv,npf_conv,sv_conv,muon_conv,gen], self.full_features)
        return full_features
    
    def predictClass(self,globalvars,cpf,npf,sv,muon,gen):
        full_features = self.extractFeatures(globalvars,cpf,npf,sv,muon,gen)
        class_prediction = self.applyLayers(full_features,self.class_prediction)
        return class_prediction
        
    def makeClassModel(self):
        gen = keras.layers.Input(shape=(len(self.featureDict["gen"]["branches"]),))
        globalvars = keras.layers.Input(shape=(len(self.featureDict["globalvars"]["branches"]),))
        cpf = keras.layers.Input(shape=(self.featureDict["cpf"]["max"], len(self.featureDict["cpf"]["branches"])))
        npf = keras.layers.Input(shape=(self.featureDict["npf"]["max"], len(self.featureDict["npf"]["branches"])))
        sv = keras.layers.Input(shape=(self.featureDict["sv"]["max"], len(self.featureDict["sv"]["branches"])))
        muon = keras.layers.Input(shape=(self.featureDict["muon"]["max"], len(self.featureDict["muon"]["branches"])))

        predictedClass = self.predictClass(globalvars,cpf,npf,sv,muon,gen)
        model = keras.models.Model(inputs=[gen,globalvars,cpf,npf,sv,muon],outputs=[predictedClass])
        return model
        
    
