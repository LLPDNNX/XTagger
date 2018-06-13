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

from plot_macros import plot_resampled, make_plots, makePlot

import llp_model_simple

# tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store', help='input file list')
parser.add_argument('--test', action='store', help='input file list')
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

arguments = parser.parse_args()

filePathTrain = arguments.train
filePathTest = arguments.test
isGPU = arguments.isGPU
nFiles = arguments.n
jobName = arguments.name
batchSize = arguments.batch
classBalance = arguments.c
num_epochs = arguments.epoch
overwriteFlag = arguments.overwriteFlag
isParametric = arguments.parametric


def print_delimiter():
    print "-"*80


# import the gpu, if needed and available
print "Trying to import the gpu, otherwise set to GPU"

print_delimiter()
if isGPU:
    try:
        if not os.environ.In('CUDA_VISIBLE_DEVICES'):
            imp.find_module('setGPU')
            import setGPU
        print "Using GPU: ", os.environ['CUDA_VISIBLE_DEVICES']
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
    except ImportError:
        print "using CPU"
        pass
else:
    print "using CPU"
print_delimiter()

fileListTrain = []
fileListTest = []

now = datetime.datetime.now()
date = str(now.year) + str(now.month) + str(now.day)
outputFolder = "output/" + date + "_" + jobName

if (os.path.exists(outputFolder) & overwriteFlag):
    print "Overwriting output folder!"
else:
    print "Creating output folder '%s'!" % outputFolder
    os.makedirs(outputFolder)
print_delimiter()

f = open(filePathTrain)
for line in f:
    if os.path.exists(line.strip()):
        fileListTrain.append(line.strip())
        print "Adding file: '"+line.strip()+"'"
    else:
        print "WARNING: file '"+line.strip()+"' does not exists -> skip!"
f.close()
print_delimiter()

f = open(filePathTest)
for line in f:
    if os.path.exists(line.strip()):
        fileListTest.append(line.strip())
        print "Adding file: '"+line.strip()+"'"
    else:
        print "WARNING: file '"+line.strip()+"' does not exists -> skip!"
f.close()
print_delimiter()

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


def setupModel(add_summary=False, options={}):
    result = {}

    # add displacement to globalvars

    # gen = keras.layers.Input(shape=(len(featureDict["gen"]["branches"]),))
    gen = keras.layers.Input(shape=(1,))
    globalvars = keras.layers.Input(
            shape=(len(featureDict["globalvars"]["branches"]),))
    cpf = keras.layers.Input(shape=(
        featureDict["cpf"]["max"], len(featureDict["cpf"]["branches"])))
    npf = keras.layers.Input(shape=(
        featureDict["npf"]["max"], len(featureDict["npf"]["branches"])))
    sv = keras.layers.Input(shape=(
        featureDict["sv"]["max"], len(featureDict["sv"]["branches"])))

    nclasses = len(featureDict["truth"]["branches"])
    print_delimiter()
    print "Nclasses =", nclasses

    # temporary fix
    if not isParametric:
        gen = None

    conv_prediction, lstm1_prediction, full_prediction = \
        llp_model_simple.model(
            globalvars, cpf, npf, sv,
            nclasses,
            gen=gen,
            options=options,
            isParametric=isParametric)

    w = 0.2*0.85**epoch
    print "Weighting loss: ", w
    print_delimiter()
    # prediction = keras.layers.Lambda(
    # lambda x: (x[0]+x[1])*0.5*w+x[2]*(1-w))
    # ([conv_prediction,lstm1_prediction,full_prediction])

    prediction = full_prediction
    if isParametric:
        return keras.Model(
                inputs=[gen, globalvars, cpf, npf, sv], outputs=prediction)

    return keras.Model(inputs=[globalvars, cpf, npf, sv], outputs=prediction)


def input_pipeline(files, batchSize):
    with tf.device('/cpu:0'):
        fileListQueue = tf.train.string_input_producer(
                files, num_epochs=1, shuffle=True)

        rootreader_op = []
        resamplers = []
        for _ in range(min(len(fileListTrain)-1, 6)):
            if isParametric:
                reader_batch = 10000
            else:
                reader_batch = 100

            reader = root_reader(fileListQueue, featureDict,
                                 "jets", batch=reader_batch).batch()

            rootreader_op.append(reader)

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

            if isParametric:

                isSignal = resampled["truth"][:, 4] > 0.5  # index 4 is LLP
                resampled["gen"] = fake_background(
                                resampled["gen"], isSignal, 0)
                print resampled["gen"]

            resamplers.append(resampled)

        minAfterDequeue = batchSize * 2
        capacity = minAfterDequeue + 3*batchSize
        batch = tf.train.shuffle_batch_join(
            # rootreader_op,
            resamplers,
            batch_size=batchSize,
            capacity=capacity,
            min_after_dequeue=minAfterDequeue,
            enqueue_many=True  # requires to read examples in batches!
        )
        return batch


learning_rate_val = 0.005
epoch = 0
previous_train_loss = 1000

while (epoch < num_epochs):

    epoch_duration = time.time()
    print_delimiter()
    print "epoch", epoch+1
    print_delimiter()

    train_batch = input_pipeline(fileListTrain, batchSize)
    test_batch = input_pipeline(fileListTest, batchSize)

    modelTrain = setupModel()
    modelTest = setupModel()

    opt = keras.optimizers.Adam(lr=learning_rate_val, beta_1=0.9, beta_2=0.999)
    modelTrain.compile(opt,
                       loss='categorical_crossentropy', metrics=['accuracy'])
    modelTest.compile(opt,
                      loss='categorical_crossentropy', metrics=['accuracy'])

    #if epoch == 0:
        #plot_model(modelTrain, to_file=os.path.join(outputFolder, 'model.eps'))

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer()
                      )

    sess = K.get_session()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    weight_path = os.path.join(
            outputFolder, "epoch_" + str(epoch-1),
            "model_epoch.hdf5")

    if os.path.exists(weight_path):
        print "loading weights ... ", weight_path
        modelTrain.load_weights(weight_path)
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
            # get the value of batch to fill histograms
            train_batch_value = sess.run(train_batch)

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

            train_outputs = modelTrain.train_on_batch(
                    train_inputs, train_batch_value["truth"])
            labelsTraining = np.add(
                    train_batch_value["truth"].sum(axis=0), labelsTraining)

            if step == 0:
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

            step += 1
            nTrainBatch = train_batch_value["truth"].shape[0]

            nTrain += nTrainBatch

            if nTrainBatch > 0:
                total_loss_train += train_outputs[0] * nTrainBatch

            if step % 10 == 0:
                duration = (time.time() - start_time)/10.
                print 'Training step %d: loss = %.3f, accuracy = %.2f%%, time = %.3f sec' % (
                    step,
                    train_outputs[0],
                    train_outputs[1]*100.,
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
      
    modelTrain.save_weights(os.path.join(outputFolder, "epoch_" + str(epoch),
                            "model_epoch.hdf5"))
    modelTest.set_weights(modelTrain.get_weights())

    hists = []
    scores = []
    truths = []

    for branches1 in featureDict["truth"]["branches"]:
        disName = branches1.replace("||", "_").replace("is", "").replace("from", "")
        histsPerDis = []
        for branches2 in featureDict["truth"]["branches"]:
            probName = branches2.replace("||", "_").replace("is", "").replace("from", "")
            h = ROOT.TH1F(disName+probName, probName, 10000, 0, 1)
            histsPerDis.append(h)
        hists.append(histsPerDis)

    try:
        step = 0
        while not coord.should_stop():
            test_batch_value = sess.run(test_batch)

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

            test_outputs = modelTest.test_on_batch(test_inputs, test_batch_value["truth"])
            test_prediction = modelTest.predict_on_batch(test_inputs)

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

            step += 1
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
