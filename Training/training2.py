import sys
import signal
import datetime
import numpy as np
import copy
import tensorflow as tf
import ROOT
import keras
from keras import backend as K
import sklearn.metrics
import logging
import math
import xtools
import os
import argparse
import time

from feature_dict import featureDict

xtools.setupLogging(level=logging.INFO)
# tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


parser = argparse.ArgumentParser()
parser.add_argument('--train', dest="trainFiles", default=[], action='append', help='input file list')
parser.add_argument('--test', dest="testFiles", default=[], action='append', help='input file list')
parser.add_argument('--trainDA', dest="trainFilesDA", default=[], action='append', help='input file list')
parser.add_argument('--testDA', dest="testFilesDA", default=[], action='append', help='input file list')
parser.add_argument('-o', '--output', action='store', help='job name', dest='outputFolder', default='')
parser.add_argument('-n', action='store', type=int, dest='maxFiles',
                    help='number of files to be processed')
parser.add_argument('--gpu', action='store_true', dest='forceGPU',
                    help='try to use gpu', default=False)
parser.add_argument('-b', '--batch', action='store', type=int,
                    help='batch_size', dest='batch_size', default=10000)
parser.add_argument('-c', action='store_true',
                    help='achieve class balance', default=False)
parser.add_argument('-e', '--epoch', action='store', type=int, dest='nepochs',
                    help='number of epochs', default=60)
parser.add_argument('-f', '--force', action='store_true',
                    dest='overwriteFlag',
                    help='overwrite output folder', default=False)
parser.add_argument('-p', '--parametric', action='store_true',
                    dest='parametric',
                    help='train a parametric model', default=False)
parser.add_argument('--opt', action='store_true',default=False,
                    dest='opt',
                    help='optimize training through back stepping on class loss')
parser.add_argument('--optDomain', action='store_true',default=False,
                    dest='optDomain',
                    help='optimize training through forward stepping on domain loss')
parser.add_argument('--noda', action='store_true',
                    dest='noda',
                    help='deactivate DA', default=False)
parser.add_argument('--wasserstein', action='store_true',
                    dest='wasserstein',
                    help='uses wasserstein distance instead', default=False)
parser.add_argument('-m', '--model', action='store', help='model file',
                    default='nominal_model')
parser.add_argument('--bagging', action='store', type=float, help='bagging fraction (default: 1. = no bagging)',
                    default=1., dest='bagging')
parser.add_argument('-r', '--resume', type=int,help='resume training at given epoch',
                    default=-1,dest='resume')
parser.add_argument('--lambda', type=float,help='domain loss weight',
                    default=0.3,dest='lambda_val')
parser.add_argument('--kappa', type=float,help='learning rate decay val',
                    default=0.1,dest='kappa')

arguments = parser.parse_args()

outputFolder = arguments.outputFolder
if (os.path.exists(outputFolder) and arguments.overwriteFlag):
    logging.warning( "Overwriting output folder!")
else:
    logging.info( "Creating output folder '%s'!" % outputFolder)
    os.makedirs(outputFolder)

devices = xtools.Devices()
if arguments.forceGPU and devices.nGPU==0:
    logging.critical("Enforcing GPU usage but no GPU found!")
    sys.exit(1)

trainInputs = xtools.InputFiles(maxFiles=arguments.maxFiles)
for f in arguments.trainFiles: 
    trainInputs.addFileList(f) 
testInputs = xtools.InputFiles(maxFiles=arguments.maxFiles)
for f in arguments.testFiles: 
    testInputs.addFileList(f) 
    
    
logging.info("Training files %i"%trainInputs.nFiles())
logging.info("Testing files %i"%testInputs.nFiles())

resampleWeights = xtools.ResampleWeights(
    trainInputs.getFileList(),
    featureDict['truth']['names'],
    featureDict['truth']['weights'],
    targetWeight='jetorigin_isLLP_QMU||jetorigin_isLLP_QQMU||jetorigin_isLLP_QQ||jetorigin_isLLP_Q',
    ptBinning=np.concatenate([np.linspace(10,40,7),np.logspace(math.log10(50),math.log10(100),4)]),
    etaBinning=np.linspace(-2.4,2.4,6)
)

resampleWeights.plot(os.path.join(outputFolder,"hists.pdf"))
weights = resampleWeights.reweight(classBalance=True,oversampling=3)
weights.plot(os.path.join(outputFolder,"weights.pdf"))
weights.save(os.path.join(outputFolder,"weights.root"))


pipelineTrain = xtools.Pipeline(
    trainInputs.getFileList(), 
    featureDict, 
    resampleWeights.getLabelNameList(),
    os.path.join(outputFolder,"weights.root"),
    arguments.batch_size
) 

pipelineTest = xtools.Pipeline(
    testInputs.getFileList(), 
    featureDict, 
    resampleWeights.getLabelNameList(),
    os.path.join(outputFolder,"weights.root"),
    arguments.batch_size
)

coord = None
def resetSession():
    if coord:
        logging.info("Stopping threads and resetting session")
        coord.request_stop()
        coord.join(threads)
        K.clear_session()

signal.signal(signal.SIGINT, lambda signum, frame: [resetSession(),sys.exit(1)])

for epoch in range(arguments.nepochs):
    start_time_epoch = time.time()
    
    network = xtools.AttentionNetwork(featureDict)
    modelClass = network.makeClassModel()
    learningRate = 0.01/(1+arguments.kappa*epoch)
    optClass = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999)
    modelClass.compile(
        optClass,
        loss=keras.losses.categorical_crossentropy,
        metrics=[keras.metrics.categorical_accuracy],
        loss_weights=[1.]
    )
    if epoch==0:
        modelClass.summary()

    train_batch = pipelineTrain.init(isLLPFct = lambda batch: (batch["truth"][:, 5]) > 0.5)
    test_batch = pipelineTest.init(isLLPFct = lambda batch: (batch["truth"][:, 5]) > 0.5)
    
    if epoch==0:
        distributions = resampleWeights.makeDistribution(np.linspace(-4,6,21))
        #featurePlotter = xtools.FeaturePlotter(featureDict)
        #preprocessingModel = network.makePreprocessingModel()

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    sess = K.get_session()
    sess.run(init_op)
    
    if epoch>0:
        networkWeightFile = os.path.join(outputFolder,'weight_%i.hdf5'%(epoch-1))
        if os.path.exists(networkWeightFile):
            logging.info("loading weights from "+networkWeightFile)
            modelClass.load_weights(networkWeightFile)
        else:
            logging.critical("No weights from previous epoch found")
            sys.exit(1)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_loss = 0    
    time_train = time.time()
    try:
        step = 0
        while not coord.should_stop():
            step += 1
            train_batch_value = sess.run(train_batch)
            if epoch==0:
                #featurePlotter.fill(train_batch_value)
                '''
                train_batch_value_preproc = preprocessingModel.predict_on_batch([
                    train_batch_value['gen'],
                    train_batch_value['globalvars'],
                    train_batch_value['cpf'],
                    train_batch_value['npf'],#
                    train_batch_value['sv'],
                    train_batch_value['muon'],
                    train_batch_value['electron'],
                ])
                featurePlotter.fill({
                    "truth":train_batch_value['truth'],
                    "gen":train_batch_value_preproc[0],
                    "globalvars":train_batch_value_preproc[1],
                    "cpf":train_batch_value_preproc[2],
                    "npf":train_batch_value_preproc[3],
                    "sv":train_batch_value_preproc[4],
                    "muon":train_batch_value_preproc[5],
                    "electron":train_batch_value_preproc[6]
                })
                '''
                distributions.fill(
                    train_batch_value['truth'],
                    train_batch_value['globalvars'][:,0],
                    train_batch_value['globalvars'][:,1],
                    train_batch_value['gen'][:,0],
                )
            #print train_batch_value
            train_inputs_class = [
                train_batch_value['gen'],
                train_batch_value['globalvars'],
                train_batch_value['cpf'],
                train_batch_value['npf'],
                train_batch_value['sv'],
                train_batch_value['muon'],
                train_batch_value['electron'],
            ]
            train_outputs = modelClass.train_on_batch(train_inputs_class,train_batch_value['truth'])
            train_loss+=train_outputs[0]
            if step%10==0:
                logging.info("Training step %i-%i: loss=%.4f, accuracy=%.2f%%"%(epoch,step,train_outputs[0],100.*train_outputs[1]))
            
            
    except tf.errors.OutOfRangeError:
        pass
        
    train_loss = train_loss/step
    time_train = (time.time()-time_train)/step
    logging.info('Done training for %i steps of epoch %i and learning rate %.4f: loss=%.3e, %.1fms/step'%(step,epoch,learningRate,train_loss,time_train*1000.))
         
    if epoch==0:   
        #featurePlotter.plot(outputFolder)
        distributions.plot(os.path.join(outputFolder,"resampled.pdf"))
    modelClass.save_weights(os.path.join(outputFolder,'weight_%i.hdf5'%epoch))
    
    
    test_loss = 0
    time_test = time.time()
    try:
        step = 0
        while not coord.should_stop():
            step += 1
            test_batch_value = sess.run(test_batch)
            #print train_batch_value
            test_inputs_class = [
                test_batch_value['gen'],
                test_batch_value['globalvars'],
                test_batch_value['cpf'],
                test_batch_value['npf'],
                test_batch_value['sv'],
                test_batch_value['muon'],
                test_batch_value['electron'],
            ]
            test_outputs = modelClass.test_on_batch(test_inputs_class,test_batch_value['truth'])
            test_loss+=test_outputs[0]
            if step%10==0:
                logging.info("Testing step %i-%i: loss=%.4f, accuracy=%.2f%%"%(epoch,step,test_outputs[0],100.*test_outputs[1]))
            
            
    except tf.errors.OutOfRangeError:
        pass
        
    test_loss = test_loss/step
    time_test = (time.time()-time_test)/step
    logging.info('Done testing for %i steps of epoch %i: loss=%.4f, %.1fms/step'%(step,epoch,test_loss,time_test*1000.))
    
    logging.info("Epoch duration: %.1fmin"%((time.time() - start_time_epoch)/60.))
    
    f = open(os.path.join(outputFolder, "model_epoch.stat"), "a")
    f.write("%i;%.3e;%.3e;%.3e\n"%(
        epoch,
        learningRate,
        train_loss,
        test_loss,
    ))
    f.close()
    
    resetSession()
    

