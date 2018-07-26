import keras
import os
import time
import math
import random
import sys
import imp
import argparse
import datetime
import numpy as np

import tensorflow as tf
import ROOT

from keras import backend as K
from keras.utils import plot_model
from xtagger import root_reader, resampler
from xtagger import classification_weights, fake_background
from sklearn.metrics import auc
from feature_dict import featureDict
from feature_dict_da import featureDictDA
from plot_macros import plot_resampled, make_plots, makePlot

# tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store', help='input file list')
parser.add_argument('--test', action='store', help='input file list')
parser.add_argument('--trainDA', action='store', help='input file list')
parser.add_argument('--testDA', action='store', help='input file list')
parser.add_argument('--name', action='store', help='job name', default='')
parser.add_argument('-n', action='store', type=int,
                    help='number of files to be processed')
parser.add_argument('--gpu', action='store_true', dest='isGPU',
                    help='try to use gpu', default=False)
parser.add_argument('-b', '--batch', action='store', type=int,
                    help='batch_size', default=10000)
parser.add_argument('-c', action='store_true',
                    help='achieve class balance', default=False)
parser.add_argument('-e', '--epoch', action='store', type=int,
                    help='number of epochs', default=60)
parser.add_argument('-o', '--overwrite', action='store_true',
                    dest='overwriteFlag',
                    help='overwrite output folder', default=False)
parser.add_argument('-p', '--parametric', action='store_true',
                    dest='parametric',
                    help='train a parametric model', default=False)
parser.add_argument('-m', '--model', action='store', help='model file',
                    default='llp_model_da')

arguments = parser.parse_args()

filePathTrain = arguments.train
filePathTest = arguments.test
filePathTrainDA = arguments.trainDA
filePathTestDA = arguments.testDA
isGPU = arguments.isGPU
nFiles = arguments.n
jobName = arguments.name
batchSize = arguments.batch
classBalance = arguments.c
num_epochs = arguments.epoch
overwriteFlag = arguments.overwriteFlag
isParametric = arguments.parametric

modelPath = arguments.model
import importlib

modelModule = importlib.import_module(modelPath)

if len(jobName)==0:
    print "Error - no job name specified"
    sys.exit(1)

def print_delimiter():
    print "-"*80

OMP_NUM_THREADS = -1
if os.environ.has_key('OMP_NUM_THREADS'):
    try:
        OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
    except Exception:
        pass



print_delimiter()

#limit CUDA_VISIBLE_DEVICES to a free GPU if unset by e.g. batch system; fails if no GPU available
try:
    if not os.environ.has_key('CUDA_VISIBLE_DEVICES'):
        imp.find_module('setGPU')
        import setGPU
        print "Using GPU: ", os.environ['CUDA_VISIBLE_DEVICES']," (manually set by 'setGPU')"
    else:
        print "Using GPU: ", os.environ['CUDA_VISIBLE_DEVICES']," (taken from env)"
except ImportError:
    pass
        
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

fileListTrain = []
fileListTest = []

now = datetime.datetime.now()
date = str(now.year) + str(now.month) + str(now.day)
outputFolder = "output/" + jobName

if (os.path.exists(outputFolder) & overwriteFlag):
    print "Overwriting output folder!"
else:
    print "Creating output folder '%s'!" % outputFolder
    os.makedirs(outputFolder)
print_delimiter()

def getListOfInputFiles(path):
    fileList = []
    f = open(path)
    for line in f:
        basepath = path.rsplit('/',1)[0]
        if os.path.exists(os.path.join(basepath,line.strip())):
            fileList.append(os.path.join(basepath,line.strip()))
            print "Adding file: '"+line.strip()+"'"
        else:
            print "WARNING: file '"+line.strip()+"' does not exists -> skip!"
    f.close()
    return fileList


fileListTrain = getListOfInputFiles(filePathTrain)
print_delimiter()
fileListTest = getListOfInputFiles(filePathTest)
print_delimiter()
fileListTrainDA = getListOfInputFiles(filePathTrainDA)
print_delimiter()
fileListTestDA = getListOfInputFiles(filePathTestDA)

# select only a fraction of files
if nFiles is not None:
    fileListTrain = fileListTrain[:nFiles]
    fileListTest = fileListTest[:nFiles]

# define the feature dictionary for training
# Count the number of entries in total
histsPerClass = {}
weightsPerClass = {}
branchNameList = []
eventsPerClass = {}
dropoutPerClass = {}
resampledEventsPerClass = {}

chain = ROOT.TChain("jets")
for f in fileListTrain:
    chain.AddFile(f)
nEntries = chain.GetEntries()
print "total entries", nEntries

# Calculate weights for resampling in pt and eta
binningPt = np.linspace(1.3, 3.0, num=30)
binningEta = np.linspace(-2.4, 2.4, num=10)
binningctau = -.5 + np.linspace(-3., 6., num=10)
targetShape = ROOT.TH2F("ptetaTarget", "", len(binningPt)-1, binningPt,
                        len(binningEta)-1, binningEta)

targetEvents = 0

print "-"*100

for label in featureDict["truth"]["branches"]:
    branchName = label.split("/")[0]
    branchNameList.append(branchName)
    print "projecting ... ", branchName
    hist = ROOT.TH2F("pteta"+branchName, "", len(binningPt)-1, binningPt,
                     len(binningEta)-1, binningEta)
    hist.Sumw2()
    chain.Project(hist.GetName(), "global_eta:global_pt",
                  "("+branchName+"==1)")
    # scale w.r.t. LLP
    if label.find("fromLLP") >= 0:
        targetShape.Add(hist)
        targetEvents += hist.GetEntries()
    if hist.Integral() > 0:
        print " -> entries ", hist.GetEntries()
        eventsPerClass[branchName] = hist.GetEntries()
    else:
        print " -> no entries found for class: ", branchName
    histsPerClass[branchName] = hist

print_delimiter()
print "class labels:", eventsPerClass.keys()
print "class balance before resampling", \
        [x / min(eventsPerClass.values()) for x in eventsPerClass.values()]
print_delimiter()

for label in branchNameList:
    hist = histsPerClass[label]
    weight = targetShape.Clone(label)
    factor = 0.
    the_sum = 0

    if (hist.Integral() > 0):
        weight.Divide(hist)
        if weight.GetMaximum() < 1:
            factor = weight.GetMaximum()
            print "rescale ", label, 1./factor
        else:
            factor = 1.

        for ibin in range(hist.GetNbinsX()+1):
            for jbin in range(hist.GetNbinsY()+1):
                if weight.GetBinContent(ibin, jbin) > 0:
                    weight.SetBinContent(ibin, jbin,
                                         weight.GetBinContent(ibin, jbin)
                                         / factor)
                    the_sum += hist.GetBinContent(ibin, jbin) * \
                        weight.GetBinContent(ibin, jbin)
                    hist.SetBinContent(ibin, jbin,
                                       targetShape.GetBinContent(ibin, jbin) /
                                       weight.GetBinContent(ibin, jbin))
                else:
                    hist.SetBinContent(ibin, jbin, 0)

    else:
        weight.Scale(0)

    resampledEventsPerClass[label] = the_sum
    weightsPerClass[label] = weight

print_delimiter()
print "class labels:", resampledEventsPerClass.keys()
print "class balance after resampling", \
    [x / min(resampledEventsPerClass.values())
        for x in resampledEventsPerClass.values()]
print_delimiter()

dropoutPerClass = {k: min(resampledEventsPerClass.values())/v
                   for k, v in resampledEventsPerClass.iteritems()}

print dropoutPerClass
print_delimiter()

weightFile = ROOT.TFile(os.path.join(outputFolder, "weights.root"), "RECREATE")
for label, hist in weightsPerClass.items():
    if classBalance:
        print "performing class balance rescaling"
        hist.Scale(dropoutPerClass[label])
    hist.Write()
weightFile.Close()

# Plot histograms of pt, eta and their weights

histsPt = {l: h.ProjectionX() for l, h in histsPerClass.items()}
histsEta = {l: h.ProjectionY() for l, h in histsPerClass.items()}

makePlot(outputFolder, histsPt, branchNameList, binningPt,
         ";Jet log(pT/1 GeV);Normalized events", "pt",
         target=targetShape.ProjectionX())
makePlot(outputFolder, histsEta, branchNameList, binningEta,
         ";Jet #eta;Normalized events", "eta",
         target=targetShape.ProjectionY())


def divide(n, d):
    r = n.Clone(d.GetName())
    r.Divide(d)
    return r


weightsPt = {l: divide(targetShape.ProjectionX(),
             h.ProjectionX()) for l, h in histsPerClass.items()}
weightsEta = {l: divide(targetShape.ProjectionY(),
              h.ProjectionY()) for l, h in histsPerClass.items()}

makePlot(outputFolder, weightsPt, branchNameList, binningPt,
         ";Jet log(pT/1 GeV);Weight", "weight_pt", logy=0)
makePlot(outputFolder, weightsEta, branchNameList, binningEta,
         ";Jet #eta;Weight", "weight_eta", logy=0)

   
def setupDiscriminators(modelDA,add_summary=False, options={}):
    result = {}

    if isParametric:
        gen = keras.layers.Input(shape=(1,))
    else:
        gen = None
    globalvars = keras.layers.Input(
            shape=(len(featureDict["globalvars"]["branches"]),))
    cpf = keras.layers.Input(shape=(
        featureDict["cpf"]["max"], len(featureDict["cpf"]["branches"])))
    npf = keras.layers.Input(shape=(
        featureDict["npf"]["max"], len(featureDict["npf"]["branches"])))
    sv = keras.layers.Input(shape=(
        featureDict["sv"]["max"], len(featureDict["sv"]["branches"])))

    classPrediction = modelDA.predictClass(globalvars,cpf,npf,sv,gen)
    domainPrediction = modelDA.predictDomain(globalvars,cpf,npf,sv,gen)

    inputs = []
    if isParametric:
        inputs.append(gen)
    inputs.append(globalvars)
    inputs.append(cpf)
    inputs.append(npf)
    inputs.append(sv)
        
    return {
        "class":
            keras.Model(
                inputs=inputs, 
                outputs=[classPrediction]
            ),
        "domain":
            keras.Model(
                inputs=inputs, 
                outputs=[domainPrediction]
            ),
        "all":
            keras.Model(
                inputs=inputs, 
                outputs=[classPrediction,domainPrediction]
            )
    }
        

def input_pipeline(files, features, batchSize, resample=True,repeat=1):
    with tf.device('/cpu:0'):
        fileListQueue = tf.train.string_input_producer(
                files, num_epochs=repeat, shuffle=True)

        rootreader_op = []
        resamplers = []
        maxThreads = 6
        if OMP_NUM_THREADS>0 and OMP_NUM_THREADS<maxThreads:
            maxThreads = OMP_NUM_THREADS
        for _ in range(min(1+int(len(fileListTrain)/2.), maxThreads)):
            reader_batch = max(10,int(batchSize/20.))
            reader = root_reader(fileListQueue, features, "jets", batch=reader_batch).batch()
            rootreader_op.append(reader)
            if resample:
                weight = classification_weights(
                    reader["truth"],
                    reader["globalvars"],
                    os.path.join(outputFolder, "weights.root"),
                    branchNameList,
                    [0, 1]
                )
                resampled = resampler(
                    weight,
                    reader
                ).resample()

                resamplers.append(resampled)

        minAfterDequeue = batchSize * 2
        capacity = minAfterDequeue + 3*batchSize
        batch = tf.train.shuffle_batch_join(
            resamplers if resample else rootreader_op,
            batch_size=batchSize,
            capacity=capacity,
            min_after_dequeue=minAfterDequeue,
            enqueue_many=True  # requires to read examples in batches!
        )
        if resample and isParametric:
            isSignal = batch["truth"][:, 4] > 0.5  # index 4 is LLP
            batch["gen"] = fake_background(batch["gen"], isSignal, 0)

        return batch


learning_rate_val = 0.005
epoch = 0
previous_train_loss = 1000

while (epoch < num_epochs):

    epoch_duration = time.time()
    print_delimiter()
    print "epoch", epoch+1
    print_delimiter()

    train_batch = input_pipeline(fileListTrain,featureDict, batchSize)
    test_batch = input_pipeline(fileListTest,featureDict, batchSize)
    
    train_batch_da = input_pipeline(fileListTrainDA,featureDictDA, batchSize,resample=False,repeat=None)
    test_batch_da = input_pipeline(fileListTestDA,featureDictDA, batchSize,resample=False,repeat=None)

    modelDA = modelModule.ModelDA(
        len(featureDict["truth"]["branches"]),
        isParametric=isParametric
    )
    
    modelDiscriminators = setupDiscriminators(modelDA)
    modelClassDiscriminator = modelDiscriminators["class"]
    modelDomainDiscriminator = modelDiscriminators["domain"]
    
    #modelTrain = setupModelDiscriminator()
    #modelTest = setupModelDiscriminator()

    optClass = keras.optimizers.Adam(lr=learning_rate_val, beta_1=0.9, beta_2=0.999)
    modelClassDiscriminator.compile(optClass,
                       loss='kullback_leibler_divergence', metrics=['accuracy'])
                       
    optDomain = keras.optimizers.Adam(lr=learning_rate_val*0.1, beta_1=0.9, beta_2=0.999)
    #alternatively: kullback_leibler_divergence, categorical_crossentropy
    
    modelDomainDiscriminator.compile(optDomain,
                       loss='binary_crossentropy', metrics=['accuracy'])
    
    if epoch == 0:
        modelClassDiscriminator.summary()
        #modelDomainDiscriminator.summary()
    
    init_op = tf.group(
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()
                    )

    sess = K.get_session()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    weight_path_class = os.path.join(
            outputFolder, "epoch_" + str(epoch-1),
            "model_epoch_class.hdf5")
    weight_path_domain = os.path.join(
            outputFolder, "epoch_" + str(epoch-1),
            "model_epoch_domain.hdf5")

    if os.path.exists(weight_path_class) and os.path.exists(weight_path_domain):
        print "loading weights ... ", weight_path_class,weight_path_domain
        modelClassDiscriminator.load_weights(weight_path_class)
        modelDomainDiscriminator.load_weights(weight_path_domain)
    elif epoch > 0:
        print "no weights from previous epoch found"
        sys.exit(1)

    # number of events
    nTrain = 0
    nTest = 0
    total_loss_train = 0
    total_loss_test = 0

    start_time = time.time()

    labelsTraining = np.array([5])

    try:
        step = 0
        while not coord.should_stop():
            step += 1
            train_batch_value = sess.run(train_batch)
            if train_batch_value['num'].shape[0]==0:
                continue
            


            if isParametric:
                train_inputs = [train_batch_value['gen'][:, 0],
                                train_batch_value['globalvars'],
                                train_batch_value['cpf'],
                                train_batch_value['npf'],
                                train_batch_value['sv']]
            else:
                train_inputs = [train_batch_value['globalvars'],
                                train_batch_value['cpf'],
                                train_batch_value['npf'],
                                train_batch_value['sv']]

            train_outputs = modelClassDiscriminator.train_on_batch(
                    train_inputs, train_batch_value["truth"])
            #train_outputs = [0,0]
            labelsTraining = np.add(
                    train_batch_value["truth"].sum(axis=0), labelsTraining)
                    
                    
            train_batch_value_domain = sess.run(train_batch_da)
            if train_batch_value_domain['num'].shape[0]>0:
                #print train_batch_value_domain["isData"][:10]
                #print train_batch_value_domain["xsecweight"][:10]
                              
                if isParametric:
                    train_inputs_domain = [np.zeros((train_batch_value_domain['num'].shape[0],1)),
                                    train_batch_value_domain['globalvars'],
                                    train_batch_value_domain['cpf'],
                                    train_batch_value_domain['npf'],
                                    train_batch_value_domain['sv']]
                else:
                    train_inputs_domain = [train_batch_value_domain['globalvars'],
                                    train_batch_value_domain['cpf'],
                                    train_batch_value_domain['npf'],
                                    train_batch_value_domain['sv']]

                train_outputs_domain = modelDomainDiscriminator.train_on_batch(
                        train_inputs_domain, train_batch_value_domain["isData"],
                        sample_weight=train_batch_value_domain["xsecweight"][:,0]
                )
               

            if step == 1:
                ptArray = train_batch_value["globalvars"][:, 0]
                etaArray = train_batch_value["globalvars"][:, 1]
                truthArray = np.argmax(train_batch_value["truth"], axis=1)
                if isParametric:
                    ctauArray = train_batch_value["gen"][:, 0]

            else:
                ptArray = np.hstack(
                        (ptArray, train_batch_value["globalvars"][:, 0]))
                etaArray = np.hstack(
                        (etaArray, train_batch_value["globalvars"][:, 1]))
                truthArray = np.hstack(
                        (truthArray,
                            np.argmax(train_batch_value["truth"], axis=1)))
                if isParametric:
                    ctauArray = np.hstack(
                            (ctauArray, train_batch_value["gen"][:, 0]))

            nTrainBatch = train_batch_value["truth"].shape[0]

            nTrain += nTrainBatch

            if nTrainBatch > 0:
                total_loss_train += train_outputs[0] * nTrainBatch

            if step % 10 == 0:
                duration = (time.time() - start_time)/10.
                print 'Training step %d: loss = %.3f (%.3f), accuracy = %.2f%% (%.2f%%), time = %.3f sec' % (
                    step,
                    train_outputs[0],
                    train_outputs_domain[0],
                    train_outputs[1]*100.,
                    train_outputs_domain[1]*100.,
                    duration
                )

                start_time = time.time()

    except tf.errors.OutOfRangeError:
        print('Done training for %d steps.' % (step))
        print_delimiter()

    if epoch == 0:

        plot_resampled(outputFolder, "pt", "$\log{(pT/1 GeV)}$",
                       ptArray, binningPt, truthArray)
        plot_resampled(outputFolder, "eta", "$\eta$",
                       etaArray, binningEta, truthArray)
        if isParametric:
            plot_resampled(outputFolder, "ctau", "$\log{(c {\\tau} / 1mm)}$",
                           ctauArray, binningctau, truthArray)

    print_delimiter()
    print "class labels:", resampledEventsPerClass.keys()
    print "predicted class balance after resampling", [x / min(resampledEventsPerClass.values()) for x in resampledEventsPerClass.values()]
    print "actual class balance after training:", labelsTraining*1./np.min(labelsTraining)
    print_delimiter()
    
    epoch_path = os.path.join(outputFolder, "epoch_" + str(epoch))
    if not (os.path.exists(epoch_path)):
      os.makedirs(os.path.join(epoch_path))
      
    modelClassDiscriminator.save_weights(os.path.join(outputFolder, "epoch_" + str(epoch),
                            "model_epoch_class.hdf5"))
    modelDomainDiscriminator.save_weights(os.path.join(outputFolder, "epoch_" + str(epoch),
                            "model_epoch_domain.hdf5"))
    hists = []
    daHists = []
    scores = []
    truths = []

    for branches1 in featureDict["truth"]["branches"]:
        disName = branches1.replace("||", "_").replace("is", "").replace("from", "")
        histsPerDis = []
        daHistsPerDis = []
        for branches2 in featureDict["truth"]["branches"]:
            probName = branches2.replace("||", "_").replace("is", "").replace("from", "")
            h = ROOT.TH1F(disName+probName, probName, 10000, 0, 1)
            h.SetDirectory(0)
            histsPerDis.append(h)
            
        hists.append(histsPerDis)
        
        daMC = ROOT.TH1F(probName+"daMC", probName, 25, 0, 1)
        daMC.SetDirectory(0)
        daMC.SetLineColor(ROOT.kAzure-4)
        daMC.SetLineWidth(3)
        daData = ROOT.TH1F(probName+"daData", probName, 25, 0, 1)
        daData.Sumw2()
        daData.SetDirectory(0)
        daData.SetMarkerStyle(20)
        daData.SetMarkerSize(1.2)
        daHists.append([daMC,daData])
    

    try:
        step = 0
        while not coord.should_stop():
            step += 1
            test_batch_value = sess.run(test_batch)
            if test_batch_value['num'].shape[0]==0:
                continue

            if isParametric:
                test_inputs = [test_batch_value['gen'][:, 0],
                               test_batch_value['globalvars'],
                               test_batch_value['cpf'],
                               test_batch_value['npf'],
                               test_batch_value['sv']]
            else:
                test_inputs = [test_batch_value['globalvars'],
                               test_batch_value['cpf'],
                               test_batch_value['npf'],
                               test_batch_value['sv']]

            test_outputs = modelClassDiscriminator.test_on_batch(test_inputs, test_batch_value["truth"])
            test_prediction = modelClassDiscriminator.predict_on_batch(test_inputs)
            
            test_batch_value_domain = sess.run(test_batch_da)
            if test_batch_value_domain['num'].shape[0]>0:
                #print train_batch_value_domain["isData"][:10]
                #print train_batch_value_domain["xsecweight"][:10]
                              
                if isParametric:
                    test_inputs_domain = [np.zeros((test_batch_value_domain['num'].shape[0],1)),
                                    test_batch_value_domain['globalvars'],
                                    test_batch_value_domain['cpf'],
                                    test_batch_value_domain['npf'],
                                    test_batch_value_domain['sv']]
                else:
                    test_inputs_domain = [test_batch_value_domain['globalvars'],
                                    test_batch_value_domain['cpf'],
                                    test_batch_value_domain['npf'],
                                    test_batch_value_domain['sv']]

                test_daprediction_class = modelClassDiscriminator.predict_on_batch(
                        test_inputs_domain
                )
                
                
                for ibatch in range(test_batch_value_domain["isData"].shape[0]):
                    isData = int(round(test_batch_value_domain["isData"][ibatch][0]))
                    sample_weight=train_batch_value_domain["xsecweight"][ibatch][0]

                    for idis in range(len(featureDict["truth"]["branches"])):
                        daHists[idis][isData].Fill(test_daprediction_class[ibatch][idis],sample_weight)

            if step == 0:
                ptArray = test_batch_value["globalvars"][:, 0]
                etaArray = test_batch_value["globalvars"][:, 1]
                truthArray = np.argmax(test_batch_value["truth"], axis=1)
                if isParametric:
                    ctauArray = test_batch_value["gen"][:, 0]

            else:
                ptArray = np.hstack((ptArray, test_batch_value["globalvars"][:, 0]))
                etaArray = np.hstack((etaArray, test_batch_value["globalvars"][:, 1]))
                truthArray = np.hstack((truthArray, np.argmax(test_batch_value["truth"], axis=1)))
                if isParametric:
                    ctauArray = np.hstack((ctauArray, test_batch_value["gen"][:, 0]))

            
            nTestBatch = test_batch_value["truth"].shape[0]

            for ibatch in range(test_batch_value["truth"].shape[0]):
                truthclass = np.argmax(test_batch_value["truth"][ibatch])
                predictedclass = np.argmax(test_prediction[ibatch])

                truths.append(truthclass)
                scores.append(predictedclass)

                for idis in range(len(featureDict["truth"]["branches"])):
                    hists[idis][truthclass].Fill(test_prediction[ibatch][idis])

            nTest += nTestBatch

            if nTestBatch > 0:
                total_loss_test += test_outputs[0] * nTestBatch

            if step % 10 == 0:
                duration = (time.time() - start_time)/10.
                print 'Testing step %d: loss = %.3f, accuracy = %.2f%%, time = %.3f sec' % ( step, test_outputs[0], test_outputs[1]*100., duration)

                start_time = time.time()

    except tf.errors.OutOfRangeError:
        print('Done testing for %d steps.' % (step))

    avgLoss_train = total_loss_train/nTrain
    avgLoss_test = total_loss_test/nTest

    if epoch == 0:

        plot_resampled(outputFolder, "testing_pt", "$\log{(p_{T} /1 GeV)}$", ptArray, binningPt, truthArray)
        plot_resampled(outputFolder, "testing_eta", "$\eta$", etaArray, binningEta, truthArray)
        if isParametric:
            plot_resampled(outputFolder, "testing_ctau", "$\log{(c {\\tau} / 1mm)}$", ctauArray, binningctau, truthArray)

    print "Epoch duration = (%.1f min)" % ((time.time() - epoch_duration)/60.)
    print "Training/Testing = %i/%i, Testing rate = %4.1f%%" % (nTrain, nTest, 100. * nTest/(nTrain+nTest))
    print "Average loss = %.4f (%.4f)" % (avgLoss_train, avgLoss_test)
    print "Learning rate = %.4e" % (learning_rate_val)

    M_score = make_plots(outputFolder, epoch, hists, truths, scores, featureDict)
    
    for idis in range(len(featureDict["truth"]["branches"])):
        probName = featureDict["truth"]["branches"][idis].replace("is", "").replace("_","").replace("from", "").replace("jetorigin","").replace(" ","")
        cv = ROOT.TCanvas("cv"+str(idis)+str(random.random()),"",800,750)
        cv.SetLogy(1)
        daHists[idis][0].Scale(daHists[idis][1].Integral()/daHists[idis][0].Integral())
        ymax = max([daHists[idis][0].GetMaximum(),daHists[idis][1].GetMaximum()])
        ymin = ymax
        for ibin in range(daHists[idis][0].GetNbinsX()):
            cMC = daHists[idis][0].GetBinContent(ibin+1)
            cData = daHists[idis][1].GetBinContent(ibin+1)
            if cMC>1 and cData>1:
                ymin = min([ymin,cMC,cData])
             
        axis = ROOT.TH2F("axis"+str(idis)+str(random.random()),";Prob("+probName+");",50,0,1,50,ymin*0.6,1.15*ymax)
        axis.Draw("AXIS")
        daHists[idis][0].Draw("HISTSAME")
        daHists[idis][1].Draw("PESAME")
        cv.Print(os.path.join(outputFolder,"epoch_" + str(epoch),"da_"+probName.replace("||","_")+".pdf"))
        cv.Print(os.path.join(outputFolder,"epoch_" + str(epoch),"da_"+probName.replace("||","_")+".png"))
    f = open(os.path.join(outputFolder, "model_epoch.stat"), "a")
    f.write(str(epoch)+";"+str(learning_rate_val)+";"+str(avgLoss_train)+";"+str(avgLoss_test)+";"+str(M_score)+"\n")
    f.close()

    if epoch > 1 and previous_train_loss < avgLoss_train:
        learning_rate_val = learning_rate_val*0.9
        print "Decreasing learning rate to %.4e" % (learning_rate_val)
    previous_train_loss = avgLoss_train

    coord.request_stop()
    coord.join(threads)
    K.clear_session()
    epoch += 1
