import keras
import os
import time
import math
import random
import sys
sys.path.append(os.path.abspath(os.path.join(sys.path[0],'../Ops')))
import imp
import argparse
import datetime

import tensorflow as tf
import numpy as np
import ROOT
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from keras import backend as K
from xtagger.root_reader import root_reader
from xtagger.resampler import resampler
from sklearn.metrics import auc

classificationweights_module = tf.load_op_library('Ops/build/libClassificationWeights.so')
fakebackground_module = tf.load_op_library('Ops/build/libFakeBackground.so')

from root_style import *
from sklearn.metrics import confusion_matrix, roc_auc_score
import llp_model_simple

# flags
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store', help='input file list')
parser.add_argument('--test', action='store', help='input file list')
parser.add_argument('--name', action='store', help='job name',default='')
parser.add_argument('-n', action='store', type=int, help='number of files to be processed')
parser.add_argument('--gpu', action='store_true', dest='isGPU', help='try to use gpu', default=False)
parser.add_argument('-b', '--batch', action='store', type=int, help='batch_size', default=10000)
parser.add_argument('-c', action='store_true', help='achieve class balance', default=False)
parser.add_argument('-e', '--epoch', action='store', type=int, help='number of epochs', default=60)
parser.add_argument('-o', '--overwrite', action='store_true', dest='overwriteFlag', help='overwrite output folder', default=False)
parser.add_argument('-p', '--parametric', action='store_true', dest='parametric', help='train a parametric model', default=False)


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

if isGPU:
    try:
        if not os.environ.has_key('CUDA_VISIBLE_DEVICES'):
            imp.find_module('setGPU')
            import setGPU
        print "Using GPU: ",os.environ['CUDA_VISIBLE_DEVICES']
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
    except ImportError:
        print "using CPU"
        pass
else:
    print "using CPU"


def plot_resampled(x, xlabel, var_array, var_binning, truth_array):

        var_0 = var_array[truth_array == 0]
        var_1 = var_array[truth_array == 1]
        var_2 = var_array[truth_array == 2]
        var_3 = var_array[truth_array == 3]
        var_4 = var_array[truth_array == 4]

        fig = plt.figure()
        plt.hist([var_0, var_1, var_2, var_3, var_4], bins=var_binning, label=['b', 'c', 'ud', 'g', 'llp'], alpha=0.5)
        plt.legend(loc='upper right')
        plt.xlabel(xlabel)
        plt.ylabel("# entries/ bin")
        plt.savefig(os.path.join(outputFolder, "reweighted_"+x+".pdf"))
        plt.close(fig)


NRGBs = 6;
NCont = 255;

stops = np.array( [0.00, 0.34,0.47, 0.61, 0.84, 1.00] )
red  = np.array( [0.5, 0.00,0.1, 1., 1.00, 0.81] )
green = np.array( [0.10, 0.71,0.85, 0.70, 0.20, 0.00] )
blue = np.array( [0.91, 1.00, 0.12,0.1, 0.00, 0.00] )

colWheelDark = ROOT.TColor.CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont)

for i in range(NRGBs):
    red[i]=min(1,red[i]*1.1+0.25)
    green[i]=min(1,green[i]*1.1+0.25)
    blue[i]=min(1,blue[i]*1.1+0.25)

colWheel = ROOT.TColor.CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont)
ROOT.gStyle.SetNumberContours(NCont)
ROOT.gRandom.SetSeed(123)
set_root_style()

fileListTrain = []
fileListTest = []

now = datetime.datetime.now()
date = str(now.year) + str(now.month) + str(now.day)
outputFolder = "output/"+date+"_"+jobName

if (os.path.exists(outputFolder) & overwriteFlag):
    print "Overwriting output folder!"
else:
    print "Creating output folder '%s'!"%outputFolder
    os.makedirs(outputFolder)

f = open(filePathTrain)
for line in f:
    if os.path.exists(line.strip()):
        fileListTrain.append(line.strip())
        print "Adding file: '"+line.strip()+"'"
    else:
        print "WARNING: file '"+line.strip()+"' does not exists -> skip!"
f.close()

f = open(filePathTest)
for line in f:
    if os.path.exists(line.strip()):
        fileListTest.append(line.strip())
        print "Adding file: '"+line.strip()+"'"
    else:
        print "WARNING: file '"+line.strip()+"' does not exists -> skip!"
f.close()

# select only a fraction of files, for debugging
if nFiles is not None:
    fileListTrain = fileListTrain[:nFiles]
    fileListTest = fileListTest[:nFiles]

# define the feature dictionary for training
featureDict = {

     "sv" : {
        "branches":[
            'sv_pt',
            'sv_deltaR',
            'sv_mass',
            'sv_ntracks',
            'sv_chi2',
            'sv_normchi2',
            'sv_dxy',
            'sv_dxysig',
            'sv_d3d',
            'sv_d3dsig',
            'sv_costhetasvpv',
            'sv_enratio',
            
        ],
        "max":4
    },

    "truth": {
        "branches":[
            'jetorigin_isB||jetorigin_isBB||jetorigin_isGBB||jetorigin_isLeptonic_B||jetorigin_isLeptonic_C',         
            'jetorigin_isC||jetorigin_isCC||jetorigin_isGCC',
            'jetorigin_isUD||jetorigin_isS',
            'jetorigin_isG',
            'jetorigin_fromLLP',
        ],
    },
    
    "gen": {
        "branches":[
            "jetorigin_ctau",
            "jetorigin_displacement"
        ]
    },
    
    "globalvars": {
        "branches": [
            'global_pt',
            'global_eta',
            'global_rho',
            'ncpf',
            'nnpf',
            'nsv',
            'csv_trackSumJetEtRatio', 
            'csv_trackSumJetDeltaR', 
            'csv_vertexCategory', 
            'csv_trackSip2dValAboveCharm', 
            'csv_trackSip2dSigAboveCharm', 
            'csv_trackSip3dValAboveCharm', 
            'csv_trackSip3dSigAboveCharm', 
            'csv_jetNSelectedTracks', 
            'csv_jetNTracksEtaRel'
        ],

    },


    "cpf": {
        "branches": [
            'cpf_trackEtaRel',
            'cpf_trackPtRel',
            'cpf_trackPPar',
            'cpf_trackDeltaR',
            'cpf_trackPParRatio',
            'cpf_trackSip2dVal',
            'cpf_trackSip2dSig',
            'cpf_trackSip3dVal',
            'cpf_trackSip3dSig',
            'cpf_trackJetDistVal',

            'cpf_ptrel', 
            'cpf_drminsv',
            'cpf_vertex_association',
            'cpf_puppi_weight',
            'cpf_track_chi2',
            'cpf_track_quality',
            #added to test
            #'cpf_jetmassdroprel',
            #'cpf_relIso01',
            #'cpf_isLepton',
            #'cpf_lostInnerHits'

        ],
        "max":25
    },
    
    "npf": {
        "branches": [
            'npf_ptrel',
            'npf_deltaR',
            'npf_isGamma',
            'npf_hcal_fraction',
            'npf_drminsv',
            'npf_puppi_weight',
            # added
            #'npf_jetmassdroprel',
            #'npf_relIso01'

        ],
        "max":25
    }
}

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
print "total entries",nEntries

# Calculate weights for resampling in pt and eta
binningPt = np.linspace(1.3,3.0,num=30)
binningEta = np.linspace(-2.4,2.4,num=10)
binningctau = -.5 + np.linspace(-3.,6.,num=10)
targetShape = ROOT.TH2F("ptetaTarget","",len(binningPt)-1,binningPt,len(binningEta)-1,binningEta)

targetEvents = 0

print "-"*100

for label in featureDict["truth"]["branches"]:
    branchName = label.split("/")[0]
    branchNameList.append(branchName)
    print "projecting ... ",branchName
    hist = ROOT.TH2F("pteta"+branchName,"",len(binningPt)-1,binningPt,len(binningEta)-1,binningEta)
    hist.Sumw2()
    chain.Project(hist.GetName(),"global_eta:global_pt","("+branchName+"==1)")
    
    # scale w.r.t. LLP
    if label.find("fromLLP")>=0:
        targetShape.Add(hist)
        targetEvents+=hist.GetEntries()
        
    if hist.Integral()>0:
        print " -> entries ",hist.GetEntries()
        eventsPerClass[branchName]=hist.GetEntries()
    else:
        print " -> no entries found for class: ",branchName
    histsPerClass[branchName]=hist
    print "-"*100

print "-"*100
print "class labels:", eventsPerClass.keys()
print "class balance before resampling", [x / min(eventsPerClass.values()) for x in eventsPerClass.values()]
print "-"*100
    
for label in branchNameList:
    hist = histsPerClass[label]
    weight = targetShape.Clone(label)
    factor = 0.
    the_sum = 0

    if (hist.Integral()>0):
        weight.Divide(hist)
        if weight.GetMaximum()<1:
            factor = weight.GetMaximum()
            print "rescale ", label,1./factor
        else:
            factor = 1.

        for ibin in range(hist.GetNbinsX()+1):
            for jbin in range(hist.GetNbinsY()+1):
                if weight.GetBinContent(ibin,jbin)>0:
                    weight.SetBinContent(ibin,jbin, weight.GetBinContent(ibin,jbin)/factor)
                    the_sum += hist.GetBinContent(ibin, jbin)*weight.GetBinContent(ibin, jbin)
                    hist.SetBinContent(ibin,jbin,
                        targetShape.GetBinContent(ibin,jbin)/weight.GetBinContent(ibin,jbin)
                    )
                else:
                    hist.SetBinContent(ibin,jbin,0)

    else:
        weight.Scale(0)

    print hist.GetEntries(), the_sum
    resampledEventsPerClass[label] = the_sum
    weightsPerClass[label]=weight

print "-"*100
print "class labels:", resampledEventsPerClass.keys()
print "class balance after resampling", [x / min(resampledEventsPerClass.values()) for x in resampledEventsPerClass.values()]
print "-"*100
 
dropoutPerClass = {k: min(resampledEventsPerClass.values())/v  for k, v in resampledEventsPerClass.iteritems()}

print "-"*100
print dropoutPerClass
print "-"*100

weightFile = ROOT.TFile(os.path.join(outputFolder,"weights.root"),"RECREATE")
for label, hist in weightsPerClass.items():
    if classBalance:
        print "performing class balance rescaling"
        hist.Scale(dropoutPerClass[label])
    hist.Write()
weightFile.Close()

# Plot histograms of pt ,eta and their weights

histsPt = {l: h.ProjectionX() for l, h in histsPerClass.items()}
histsEta = {l: h.ProjectionY() for l, h in histsPerClass.items()}

makePlot(outputFolder, histsPt, branchNameList, binningPt,";Jet log(pT/1 GeV);Normalized events","pt",target=targetShape.ProjectionX())
makePlot(outputFolder, histsEta, branchNameList, binningEta,";Jet #eta;Normalized events","eta",target=targetShape.ProjectionY())

def divide(n,d):
    r = n.Clone(d.GetName())
    r.Divide(d)
    return r
weightsPt = {l: divide(targetShape.ProjectionX(),h.ProjectionX()) for l, h in histsPerClass.items()}
weightsEta = {l: divide(targetShape.ProjectionY(),h.ProjectionY()) for l, h in histsPerClass.items()}

makePlot(outputFolder, weightsPt,branchNameList,binningPt,";Jet log(pT/1 GeV);Weight","weight_pt",logy=0)
makePlot(outputFolder, weightsEta,branchNameList,binningEta,";Jet #eta;Weight","weight_eta",logy=0)

def setupModel(add_summary=False,options={}):
    result = {}

    # add displacement to globalvars

    #gen = keras.layers.Input(shape=(len(featureDict["gen"]["branches"]),))
    gen = keras.layers.Input(shape=(1,))
    globalvars = keras.layers.Input(shape=(len(featureDict["globalvars"]["branches"]),))
    cpf = keras.layers.Input(shape=(featureDict["cpf"]["max"],len(featureDict["cpf"]["branches"])))
    npf = keras.layers.Input(shape=(featureDict["npf"]["max"],len(featureDict["npf"]["branches"])))
    sv = keras.layers.Input(shape=(featureDict["sv"]["max"],len(featureDict["sv"]["branches"])))

    print gen
    print globalvars
    print cpf
    print npf
    print sv

    nclasses = len(featureDict["truth"]["branches"])
    print "Nclasses = ",nclasses
    
    #temporary fix
    if not isParametric:
        gen=None
        
    conv_prediction,lstm1_prediction,full_prediction = llp_model_simple.model(
        globalvars,cpf,npf,sv,
        nclasses,
        gen=gen,
        options=options,
        isParametric=isParametric
    )

    w = 0.2*0.85**epoch
    print "Weighting loss: ",w
    #prediction = keras.layers.Lambda(lambda x: (x[0]+x[1])*0.5*w+x[2]*(1-w))([conv_prediction,lstm1_prediction,full_prediction])
    prediction = full_prediction
    if isParametric:
        return keras.Model(inputs=[gen,globalvars,cpf,npf,sv], outputs=prediction)
    
    return keras.Model(inputs=[globalvars,cpf,npf,sv], outputs=prediction)
    
def input_pipeline(files, batchSize):
    with tf.device('/cpu:0'):
        fileListQueue = tf.train.string_input_producer(files, num_epochs=1, shuffle=True)

        rootreader_op = []
        resamplers = []
        for _ in range(min(len(fileListTrain)-1,6)):
            reader = root_reader(fileListQueue, featureDict,"jets",batch=10000).batch() 
            rootreader_op.append(reader)
            
            weight = classificationweights_module.classification_weights(
                reader["truth"],
                reader["globalvars"],
                os.path.join(outputFolder,"weights.root"),
                branchNameList,
                [0,1]
            )
            resampled = resampler(
                weight,
                reader
            ).resample()
            
            if isParametric:

                isSignal = resampled["truth"][:,4]>0.5 #index 4 is LLP
                resampled["gen"] = fakebackground_module.fake_background(resampled["gen"],isSignal,0)
                print resampled["gen"]
            
            resamplers.append(resampled)
        
        minAfterDequeue = batchSize*2
        capacity = minAfterDequeue + 3*batchSize
        
        batch = tf.train.shuffle_batch_join(
            #rootreader_op, 
            resamplers,
            batch_size=batchSize, 
            capacity=capacity,
            min_after_dequeue=minAfterDequeue,
            enqueue_many=True #requires to read examples in batches!
        )
        return batch
    
learning_rate_val = 0.005
epoch = 0
previous_train_loss = 1000

# Create root file to hold resampled pt, eta, lifetime values
#t_resampled = ROOT.TTree("resampled","resampled")
#binningctau = np.linspace(-3.,5.,9)
#pt_hist = ROOT.TH1F("pt","",len(binningPt)-1,binningPt)
#eta_hist = ROOT.TH1F("eta","",len(binningEta)-1,binningEta)
#ctau_hist = ROOT.TH1F("ctau","",len(binningctau)-1,binningctau)

while (epoch<num_epochs):

    epoch_duration = time.time()
    print "epoch",epoch+1
    
    train_batch = input_pipeline(fileListTrain, batchSize)
    test_batch = input_pipeline(fileListTest, batchSize)

    modelTrain = setupModel()
    modelTest = setupModel()
    
    opt = keras.optimizers.Adam(lr=learning_rate_val, beta_1=0.9, beta_2=0.999)
    modelTrain.compile(opt,loss='categorical_crossentropy',metrics=['accuracy'])
    modelTest.compile(opt,loss='categorical_crossentropy',metrics=['accuracy'])
    modelTrain.summary()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess = K.get_session()
    sess.run(init_op)
    
    #summary_writer = tf.summary.FileWriter(os.path.join(outputFolder,"log"+str(epoch)), sess.graph)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    if os.path.exists(os.path.join(outputFolder,"model_epoch"+str(epoch-1)+".hdf5")):
        print "loading weights ... ",os.path.join(outputFolder,"model_epoch"+str(epoch-1)+".hdf5")
        modelTrain.load_weights(os.path.join(outputFolder,"model_epoch"+str(epoch-1)+".hdf5"))
    elif epoch>0:
        print "no weights from previous epoch found"
        sys.exit(1)
        
    total_loss_train = 0
    total_loss_test = 0
    
    # number of events
    nTrain = 0
    nTest = 0
    start_time = time.time()

    #histsPerClassTraining = {}
    #hist = ROOT.TH2F("pteta"+branchName,"",len(binningPt)-1,binningPt,len(binningEta)-1,binningEta)
    #hist.Sumw2()
    
    #hist.Smooth()

    labelsTraining = np.array([5])

    try:
        step = 0
        while not coord.should_stop():
            # get the value of batch to fill histograms
            train_batch_value = sess.run(train_batch)

            if isParametric:
                train_inputs = [train_batch_value['gen'][:,0],
                    train_batch_value['globalvars'],
                    train_batch_value['cpf'],
                    train_batch_value['npf'],
                    train_batch_value['sv']
                        ]
            else:
                train_inputs = [train_batch_value['globalvars'],
                    train_batch_value['cpf'],
                    train_batch_value['npf'],
                    train_batch_value['sv']
                        ]

            train_outputs = modelTrain.train_on_batch(train_inputs, train_batch_value["truth"])
            labelsTraining = np.add(train_batch_value["truth"].sum(axis=0), labelsTraining)

            if step == 0:
                ptArray = train_batch_value["globalvars"][:,0]
                etaArray = train_batch_value["globalvars"][:,1]
                truthArray = np.argmax(train_batch_value["truth"], axis=1)
                if isParametric:
                    ctauArray = train_batch_value["gen"][:,0]

            else: 
                ptArray = np.hstack((ptArray, train_batch_value["globalvars"][:,0]))
                etaArray = np.hstack((etaArray, train_batch_value["globalvars"][:,1]))
                truthArray = np.hstack((truthArray, np.argmax(train_batch_value["truth"], axis=1)))
                if isParametric:
                    ctauArray = np.hstack((ctauArray,train_batch_value["gen"][:,0]))

            step += 1
            nTrainBatch = train_batch_value["truth"].shape[0]

            nTrain += nTrainBatch
            
            if nTrainBatch>0:
                total_loss_train+=train_outputs[0]*nTrainBatch
                         
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

    if epoch == 0:

        plot_resampled("pt", "$\log{(pT/1 GeV)}$", ptArray, binningPt, truthArray)
        plot_resampled("eta", "$\eta$", etaArray, binningEta, truthArray)
        if isParametric:
            plot_resampled("ctau", "$\log{(c {\\tau} / 1mm)}$", ctauArray, binningctau, truthArray)

    print "-"*100
    print "class labels:", resampledEventsPerClass.keys()
    print "predicted class balance after resampling", [x / min(resampledEventsPerClass.values()) for x in resampledEventsPerClass.values()]
    print "actual class balance after training:", labelsTraining*1./np.min(labelsTraining)
    print "-"*100
 
    modelTrain.save_weights(os.path.join(outputFolder,"model_epoch"+str(epoch)+".hdf5"))
    modelTest.set_weights(modelTrain.get_weights())

    hists = []
    
    scores = []
    truths = []
    
    for branches1 in featureDict["truth"]["branches"]:
        disName = branches1.replace("||","_").replace("is","").replace("from","")
        histsPerDis = []
        for branches2 in featureDict["truth"]["branches"]:
            probName = branches2.replace("||","_").replace("is","").replace("from","")
            h = ROOT.TH1F(disName+probName,probName,10000,0,1)
            histsPerDis.append(h)
        hists.append(histsPerDis)
        
    try:
        step = 0
        while not coord.should_stop():
            test_batch_value = sess.run(test_batch)
            
            if isParametric:
                test_inputs = [test_batch_value['gen'][:,0],
                    test_batch_value['globalvars'],
                    test_batch_value['cpf'],
                    test_batch_value['npf'],
                    test_batch_value['sv']
                        ]
            else:
                test_inputs = [test_batch_value['globalvars'],
                    test_batch_value['cpf'],
                    test_batch_value['npf'],
                    test_batch_value['sv']
                        ]

            test_outputs = modelTest.test_on_batch(test_inputs, test_batch_value["truth"])
            test_prediction = modelTest.predict_on_batch(test_inputs)

            if step == 0:
                ptArray = test_batch_value["globalvars"][:,0]
                etaArray = test_batch_value["globalvars"][:,1]
                truthArray = np.argmax(test_batch_value["truth"], axis=1)
                if isParametric:
                    ctauArray = test_batch_value["gen"][:,0]


            else: 
                ptArray = np.hstack((ptArray, test_batch_value["globalvars"][:,0]))
                etaArray = np.hstack((etaArray, test_batch_value["globalvars"][:,1]))
                truthArray = np.hstack((truthArray, np.argmax(test_batch_value["truth"], axis=1)))
                if isParametric:
                    ctauArray = np.hstack((ctauArray,test_batch_value["gen"][:,0]))

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
            
            if nTestBatch>0:
                total_loss_test+=test_outputs[0]*nTestBatch
                         
            if step % 10 == 0:
                duration = (time.time() - start_time)/10.
                print 'Testing step %d: loss = %.3f, accuracy = %.2f%%, time = %.3f sec' % (
                    step,
                    test_outputs[0],
                    test_outputs[1]*100.,
                    duration
                )
                
                start_time = time.time()
            
    except tf.errors.OutOfRangeError:
        print('Done testing for %d steps.' % (step))

    avgLoss_train = total_loss_train/nTrain
    avgLoss_test = total_loss_test/nTest

    if epoch == 0:

        plot_resampled("testing_pt", "$\log{(p_{T} /1 GeV)}$", ptArray, binningPt, truthArray)
        plot_resampled("testing_eta", "$\eta$", etaArray, binningEta, truthArray)
        if isParametric:
            plot_resampled("testing_ctau", "$\log{(c {\\tau} / 1mm)}$", ctauArray, binningctau, truthArray)


    print "Epoch duration = (%.1f min)"%((time.time()-epoch_duration)/60.)
    print "Training/Testing = %i/%i, Testing rate = %4.1f%%"%(nTrain,nTest,100.*nTest/(nTrain+nTest))
    print "Average loss = %.4f (%.4f)"%(avgLoss_train,avgLoss_test)
    print "Learning rate = %.4e"%(learning_rate_val)
    
    names = [
        "b jet",
        "c jet",
        "uds jet",
        "g jet",
        "LLP jet"
    ]
    
    M_score = 0
    dimension = len(featureDict["truth"]["branches"])
    all_aucs = np.zeros([dimension,dimension])
    for truth_label in range(dimension):
        signalHist = hists[truth_label][truth_label]
        rocs = []
        name = []
        aucs = []

        for prob_label in range(dimension):
            if truth_label==prob_label:
                continue
            bkgHist = hists[truth_label][prob_label]
            sigEff, bgRej, bgEff = getROC(signalHist,bkgHist)
            auc = getAUC(sigEff,bgRej) + 0.5
            length = len(sigEff)
            sigEff = np.array(sigEff)
            bgEff = np.array(bgEff)
            auc2 = np.trapz(sigEff,bgEff)
            all_aucs[prob_label,truth_label] = auc2
            aucs.append(auc2)
            print "truth: (signal) ", names[truth_label], "prob: (background) ", names[prob_label], "auc: ", auc , "auc2: ", auc2
            
            graph = ROOT.TGraph(len(sigEff), sigEff, bgEff)
            graph.SetLineWidth(3)
            graph.SetLineStyle(1+prob_label%2)
            graph.SetLineColor(int(colWheelDark+250.*prob_label/(len(featureDict["truth"]["branches"])-1)))
            rocs.append(graph)
            name.append(names[prob_label])

        cv = ROOT.TCanvas("cv_roc"+str(prob_label),"",800,600)
        cv.SetRightMargin(0.25)
        cv.SetBottomMargin(0.18)
        cv.SetLeftMargin(0.16)
        cv.SetLogy(1)
        axis=ROOT.TH2F("axis"+str(random.random()),";"+names[truth_label]+" efficiency;Background efficiency",50,0,1.0,50,0.00008,1.0)
        axis.GetYaxis().SetNdivisions(508)
        axis.GetXaxis().SetNdivisions(508)
        axis.GetXaxis().SetTickLength(0.015/(1-cv.GetLeftMargin()-cv.GetRightMargin()))
        axis.GetYaxis().SetTickLength(0.015/(1-cv.GetTopMargin()-cv.GetBottomMargin()))
        #axis.GetYaxis().SetNoExponent(True)
        axis.Draw("AXIS")
        legend = ROOT.TLegend(0.76,0.9,0.99,0.25)
        legend.SetBorderSize(0)
        legend.SetTextFont(42)
        legend.SetFillStyle(0)
        
        for prob_label,roc in enumerate(rocs):
            roc.Draw("SameL")
            legend.AddEntry(roc,name[prob_label],"L")
            legend.AddEntry("","AUC %.1f%%"%(aucs[prob_label]*100.),"")
        legend.Draw("Same")
        
        cv.Print(os.path.join(outputFolder,"roc "+names[truth_label]+" epoch"+str(epoch)+".pdf"))
        cv.Print(os.path.join(outputFolder,"roc "+names[truth_label]+" epoch"+str(epoch)+".root"))

        for prob_label in range(truth_label):
            average_auc = .5*(all_aucs[prob_label,truth_label] + all_aucs[truth_label, prob_label])
            print "average auc between ", names[truth_label]," and ", names[prob_label]," is: ", average_auc
            M_score += average_auc
            
    M_score = 2.*M_score/((dimension)*(dimension-1)) 
    print "-"*100
    print "The M score is: ", M_score
    print "-"*100
    
    rootOutput = ROOT.TFile(os.path.join(outputFolder,"report_epoch"+str(epoch)+".root"),"RECREATE")
    
    for idis in range(dimension):
        cv = ROOT.TCanvas("cv"+str(idis),"",800,600)
        cv.SetRightMargin(0.25)
        cv.SetBottomMargin(0.18)
        cv.SetLeftMargin(0.16)
        
        ymax = 0.
        ymin = 1000
        for h in hists[idis]:
            h.Rebin(200)
            if h.Integral()>0:
                h.SetDirectory(rootOutput)
                h.Scale(1./h.Integral())
                ymax = max(ymax,h.GetMaximum())
                ymin = max(10**-5.5,min(ymin,h.GetMinimum()))
        disName = names[idis]
        axis = ROOT.TH2F("axis"+str(idis),";Prob("+disName+");Normalized events",50,0,1,50,ymin*0.6,ymax*1.2)
        axis.Draw("AXIS")
        cv.SetLogy(1)
        legend = ROOT.TLegend(0.76,0.9,0.99,0.35)
        legend.SetBorderSize(0)
        legend.SetTextFont(42)
        legend.SetFillStyle(0)
        
        for iprob in range(dimension):
            hists[idis][iprob].SetLineColor(int(colWheelDark+250.*iprob/(len(featureDict["truth"]["branches"])-1)))
            hists[idis][iprob].SetLineWidth(3)
            hists[idis][iprob].SetLineStyle(1+iprob%2)
            hists[idis][iprob].Draw("HISTSame")
            legend.AddEntry(hists[idis][iprob],names[iprob],"L")
        legend.Draw("Same")
        cv.Print(os.path.join(outputFolder,disName+" epoch"+str(epoch)+".pdf"))
        #cv.Print(os.path.join(outputFolder,disName+" epoch"+str(epoch)+".png"))
        
    conf_matrix = confusion_matrix(
        y_true=np.array(truths,dtype=int), 
        y_pred=np.array(scores,dtype=int),
        labels=range(len(featureDict["truth"]["branches"]))
    )


    
    conf_matrix_norm = np.zeros(conf_matrix.shape)
    for itruth in range(len(featureDict["truth"]["branches"])):
        total = 0.0
        for ipred in range(len(featureDict["truth"]["branches"])):
            total += conf_matrix[itruth][ipred]
        for ipred in range(len(featureDict["truth"]["branches"])):
            conf_matrix_norm[itruth][ipred] = 1.*conf_matrix[itruth][ipred]/total
      
    hist_conf = ROOT.TH2F("conf_hist","",
        len(featureDict["truth"]["branches"]),0,len(featureDict["truth"]["branches"]),
        len(featureDict["truth"]["branches"]),0,len(featureDict["truth"]["branches"])
    )

    hist_conf.SetDirectory(rootOutput)
    for itruth in range(len(featureDict["truth"]["branches"])):
        hist_conf.GetYaxis().SetBinLabel(itruth+1,"Pred. "+names[itruth])
        hist_conf.GetXaxis().SetBinLabel(itruth+1,"True "+names[itruth])
        for ipred in range(len(featureDict["truth"]["branches"])):
            hist_conf.SetBinContent(itruth+1,ipred+1,conf_matrix_norm[itruth][ipred]*100.)
    hist_conf.GetZaxis().SetTitle("Accuracy (%)")
    hist_conf.GetXaxis().SetLabelOffset(0.02)
    hist_conf.GetZaxis().SetTitleOffset(1.25)
    hist_conf.SetMarkerSize(1.8)
    cv = ROOT.TCanvas("conf","",900,700)
    cv.SetRightMargin(0.22)
    cv.SetBottomMargin(0.18)
    cv.SetLeftMargin(0.25)
    hist_conf.Draw("colztext")
    cv.Print(os.path.join(outputFolder,"confusion epoch"+str(epoch)+".pdf"))
    f = open(os.path.join(outputFolder,"model_epoch.stat"),"a")
    f.write(str(epoch)+";"+str(learning_rate_val)+";"+str(avgLoss_train)+";"+str(avgLoss_test)+";"+str(M_score)+"\n")
    f.close()
    rootOutput.Write()
    rootOutput.Close()
    if epoch>2 and previous_train_loss<avgLoss_train:
        learning_rate_val = learning_rate_val*0.9
        print "Decreasing learning rate to %.4e"%(learning_rate_val)
    previous_train_loss = avgLoss_train
 
    coord.request_stop()
    coord.join(threads)
    K.clear_session()
    
    epoch+=1

