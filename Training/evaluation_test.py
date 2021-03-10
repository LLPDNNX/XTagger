import os
import time
import h5py
import math
import random
import sys
import imp
import argparse
import datetime
import numpy as np
import copy

os.environ['KERAS_BACKEND']='tensorflow'
import tensorflow as tf
import keras
import ROOT

from keras import backend as K
from keras.utils import plot_model
from sklearn.metrics import auc
from feature_dict import featureDict as featureDictTmpl
from plot_macros import plot_resampled, make_plots, makePlot
from style import ctauSymbol

def print_delimiter():
    print "-"*80

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        i = 0
        try:
            print l.output_shape
        except AttributeError:
              print("An exception occurred")
              continue

        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem


    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes

# tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store', help='input file list')
parser.add_argument('--name', action='store', help='job name', default='')
parser.add_argument('-n', action='store', type=int,
                    help='number of files to be processed')
parser.add_argument('--gpu', action='store_true', dest='isGPU',
                    help='try to use gpu', default=False)
parser.add_argument('-o', '--overwrite', action='store_true',
                    dest='overwriteFlag',
                    help='overwrite output folder', default=False)
parser.add_argument('-m', '--model', action='store', help='model file',
                    default='nominal_model')
parser.add_argument('-w', '--weight', action='store', help='weights for the model', default='model_epoch_class.hdf5')

arguments = parser.parse_args()

filePathTest = arguments.test
isGPU = arguments.isGPU
nFiles = arguments.n
jobName = arguments.name
overwriteFlag = arguments.overwriteFlag
weight_path_class = arguments.weight

modelPath = arguments.model
import importlib
modelModule = importlib.import_module(modelPath)

if len(jobName)==0:
    print "Error - no job name specified"
    sys.exit(1)


OMP_NUM_THREADS = -1
if os.environ.has_key('OMP_NUM_THREADS'):
    try:
        OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
    except Exception:
        pass

featureDict = copy.deepcopy(featureDictTmpl)

print_delimiter()
        
#query all available devices
nCPU = 0
nGPU = 0
nUnknown = 0
from tensorflow.python.client import device_lib
for dev in device_lib.list_local_devices():
    if dev.device_type=="CPU":
        nCPU+=1
    elif dev.device_type=="GPU":
        nGPU+=1
    else:
        nUnknown+=1
print "Found %i/%i/%i (CPU/GPU/unknown) devices"%(nCPU,nGPU,nUnknown)
        
if isGPU and nGPU==0:
    print "Enforcing gpu usage"
    raise Exception("GPU usage enforced but no GPU found or available for computation")
print_delimiter()

fileListTest = []

now = datetime.datetime.now()
date = str(now.year) + str(now.month) + str(now.day)
outputFolder = "output/" + jobName

if (os.path.exists(outputFolder) and (overwriteFlag)):
    print "Overwriting output folder!"
else:
    print "Creating output folder '%s'!" % outputFolder
    os.makedirs(outputFolder)
print_delimiter()

def getListOfInputFiles(path):
    fileList = []
    f = open(path)
    for line in f:
        if os.path.exists(os.path.join(line.strip())):
            fileList.append(os.path.join(line.strip()))
            print "Adding file: '"+line.strip()+"'"
        else:
            print "WARNING: file '"+line.strip()+"' does not exists -> skip!"
    f.close()
    return fileList

fileListTest = getListOfInputFiles(filePathTest)
print_delimiter()

# select only a fraction of files
if (nFiles is not None) and (nFiles<len(fileListTest)):
    fileListTest = fileListTest[:nFiles]
    
   
def setupDiscriminators(modelDA,add_summary=False, options={}):
    result = {}

    gen = keras.layers.Input(shape=(1,))
    globalvars = keras.layers.Input(
            shape=(len(featureDict["globalvars"]["branches"]),))
    cpf = keras.layers.Input(shape=(
        featureDict["cpf"]["max"], len(featureDict["cpf"]["branches"])))
    npf = keras.layers.Input(shape=(
        featureDict["npf"]["max"], len(featureDict["npf"]["branches"])))
    sv = keras.layers.Input(shape=(
        featureDict["sv"]["max"], len(featureDict["sv"]["branches"])))

    classPrediction = modelDA.predictClass(globalvars,cpf,npf,sv,gen)

    inputs = []
    inputs.append(gen)
    inputs.append(globalvars)
    inputs.append(cpf)
    inputs.append(npf)
    inputs.append(sv)
        
    return keras.Model(
            inputs=inputs, 
            outputs=[classPrediction])
        

def load_batch(i, fileList):
    if (i >= len(fileList)):
        return -1
    fileName = fileList[i]
    groups = {}
    with h5py.File(fileName, 'r') as hf:
        for group in ['cpf', 'npf', 'sv', 'gen', 'globalvars', 'truth']:
            groups[group] = hf.get(group).value
    return groups


def random_ctau(start,end,v):
    #use pseudo random hash
    h = ((v >> 16) ^ v) * 0x45d9f3b
    h = ((h >> 16) ^ h) * 0x45d9f3b
    h = (h >> 16) ^ h
    return start+((17+h+h/100+h/10000)%(end-start))
    

epoch_duration = time.time()
print_delimiter()

modelDA = modelModule.ModelDA(
    len(featureDict["truth"]["branches"]),
    isParametric=True,
    useLSTM=False,
    useWasserstein=False
)

modelClassDiscriminator = setupDiscriminators(modelDA)

print "class network"
modelClassDiscriminator.summary()

init_op = tf.group(
    tf.global_variables_initializer(),
    tf.local_variables_initializer()
)

sess = K.get_session()
sess.run(init_op)
flops = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
print('FLOP = ', flops.total_float_ops)

print(modelClassDiscriminator)

if os.path.exists(weight_path_class):
    print "loading weights ", weight_path_class#,weight_path_domain
    modelClassDiscriminator.load_weights(weight_path_class)
else:
    print "no weights found"
    sys.exit(1)

# number of events
nTest = 0

start_time = time.time()
labelsTraining = np.array([5])

ptArray = []
etaArray = []
truthArray = []
ctauArray = []

step = 0
while True:
    test_batch_value = load_batch(step, fileListTest)
    if test_batch_value == -1:
        break
    step += 1

    test_inputs = [test_batch_value['gen'],
                    test_batch_value['globalvars'],
                    test_batch_value['cpf'],
                    test_batch_value['npf'],
                    test_batch_value['sv']]

    #test_outputs = modelClassDiscriminator.test_on_batch(test_inputs, test_batch_value["truth"])
    test_prediction = modelClassDiscriminator.predict_on_batch(test_inputs)
    
    nTestBatch = test_batch_value["truth"].shape[0]

    for ibatch in range(test_batch_value["truth"].shape[0]):
        truthclass = np.argmax(test_batch_value["truth"][ibatch])
        predictedclass = np.argmax(test_prediction[ibatch])

        # truths.append(truthclass)
        # scores.append(predictedclass)

    nTest += nTestBatch
    
    #if nTestBatch>0:
        #total_loss_test += test_outputs[0]*nTestBatch#/classLossWeight

    if step % 10 == 0:
        duration = (time.time() - start_time)/10.
        # print 'Testing step %d: loss = %.3f, accuracy = %.2f%%, time = %.3f sec' % ( step, test_outputs[0], test_outputs[1]*100., duration)
        print 'Testing step %d: time = %.3f sec' % ( step, duration)

        start_time = time.time()

print('Done testing for %d steps.' % (step))