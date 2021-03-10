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
parser.add_argument('-b', '--batch', action='store', type=int,
                    help='batch_size', default=1000)
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
batchSize = arguments.batch
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

fileListTrain = []
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
print get_model_memory_usage(batchSize, modelClassDiscriminator)

init_op = tf.group(
    tf.global_variables_initializer(),
    tf.local_variables_initializer()
)

sess = K.get_session()
sess.run(init_op)
flops = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
print('FLOP = ', flops.total_float_ops)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
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

try:
    step = 0
    while not coord.should_stop():
        test_batch_value = load_batch(step, fileListTrain)
        if test_batch_value == -1:
            break
        step += 1

        if isParametric:
            test_inputs = [test_batch_value['gen'],
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
        
        #print train_batch_value_domain["isData"][:10]
        #print train_batch_value_domain["xsecweight"][:10]  

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
        
        if nTestBatch>0:
            total_loss_test += test_outputs[0]*nTestBatch#/classLossWeight

        if step % 10 == 0:
            duration = (time.time() - start_time)/10.
            print 'Testing step %d: loss = %.3f, accuracy = %.2f%%, time = %.3f sec' % ( step, test_outputs[0], test_outputs[1]*100., duration)

            start_time = time.time()

except tf.errors.OutOfRangeError:
    print('Done testing for %d steps.' % (step))
    
# if not noDA:
#     try:
#         step = 0
#         while not coord.should_stop():
#             step += 1
#             test_batch_value_domain = sess.run(test_batch_da)
#             if test_batch_value_domain['num'].shape[0]==0:
#                 continue
#             for logctau in range(-2, 5):
#                 if isParametric:
#                     test_inputs_domain = [np.ones((test_batch_value_domain['num'].shape[0],1))*logctau,
#                                     test_batch_value_domain['globalvars'],
#                                     test_batch_value_domain['cpf'],
#                                     test_batch_value_domain['npf'],
#                                     test_batch_value_domain['sv']]
#                 else:
#                     test_inputs_domain = [test_batch_value_domain['globalvars'],
#                                     test_batch_value_domain['cpf'],
#                                     test_batch_value_domain['npf'],
#                                     test_batch_value_domain['sv']]

#                 test_outputs_domain = modelDomainDiscriminator.test_on_batch(
#                         test_inputs_domain,
#                         (2.*test_batch_value_domain["isData"]-1) if useWasserstein else test_batch_value_domain["isData"],
#                         sample_weight=test_batch_value_domain["xsecweight"][:,0]
#                 )
#                 test_daprediction_class = modelClassDiscriminator.predict_on_batch(
#                         test_inputs_domain
#                 )
                
#                 for ibatch in range(test_batch_value_domain["isData"].shape[0]):
#                     isData = int(round(test_batch_value_domain["isData"][ibatch][0]))
#                     sample_weight=test_batch_value_domain["xsecweight"][ibatch][0]

#                     for idis in range(len(featureDict["truth"]["branches"])):
#                         daHists[logctau][idis][isData].Fill(test_daprediction_class[ibatch][idis],sample_weight)

                

#                 if logctau == 0:

#                     nTestBatchDomain = test_batch_value_domain["isData"].shape[0]

#                     nTestDomain += nTestBatchDomain

#                     if nTestBatchDomain>0:
#                         total_loss_test_domain += test_outputs_domain[0] * nTestBatchDomain#/domainLossWeight

#                     if step % 10 == 0:
#                         duration = (time.time() - start_time)/10.
#                         print 'Testing DA step %d: loss = %.3f, accuracy = %.2f%%, time = %.3f sec' % ( step, test_outputs_domain[0], test_outputs_domain[1]*100., duration)

#                         start_time = time.time()

#     except tf.errors.OutOfRangeError:
#         print('Done testing for %d steps.' % (step))

# avgLoss_train = total_loss_train/nTrain
# avgLoss_test = total_loss_test/nTest
# if noDA:
#     avgLoss_train_domain = 0
#     avgLoss_test_domain = 0
# else:
#     avgLoss_train_domain = total_loss_train_domain/nTrainDomain
#     avgLoss_test_domain = total_loss_test_domain/nTestDomain
    
# avgLoss_train_per_epoch.append(avgLoss_train)
# avgLoss_test_per_epoch.append(avgLoss_test)
# avgLoss_train_domain_per_epoch.append(avgLoss_train_domain)
# avgLoss_test_domain_per_epoch.append(avgLoss_test_domain)

# if epoch == 0:

#     plot_resampled(outputFolder, "testing_pt", "$\log{(p_{T} /1 GeV)}$", ptArray, binningPt, truthArray)
#     plot_resampled(outputFolder, "testing_eta", "$\eta$", etaArray, binningEta, truthArray)
#     if isParametric:
#         plot_resampled(outputFolder, "testing_ctau", "$\log{(c {\\tau} / 1mm)}$", ctauArray, binningctau, truthArray)

# print "Epoch duration = (%.1f min)" % ((time.time() - epoch_duration)/60.)
# print "Training/Testing = %i/%i, Testing rate = %4.1f%%" % (nTrain, nTest, 100. * nTest/(nTrain+nTest))
# print "Average loss class = %.4f (%.4f)" % (avgLoss_train, avgLoss_test)
# if not noDA:
#     print "Average loss domain = %.4f (%.4f)" % (avgLoss_train_domain, avgLoss_test_domain)
# print "Learning rate = %.4e" % (learning_rate_val)

# tight_WP_effs, M_score = make_plots(outputFolder, epoch, hists, truths, scores, featureDict)

# labels = ["B","C","UDS","G","LLP"]
# KS_scores = []

# if not noDA:
#     for logctau in range(-2, 4):
#         for idis in range(len(featureDict["truth"]["branches"])):
#             cv = ROOT.TCanvas("cv"+str(idis)+str(random.random()),"",800,750)
#             cv.Divide(1,2,0,0)
#             cv.GetPad(1).SetPad(0.0, 0.0, 1.0, 1.0)
#             cv.GetPad(2).SetPad(0.0, 0.0, 1.0, 1.0)
#             cv.GetPad(1).SetFillStyle(4000)
#             cv.GetPad(2).SetFillStyle(4000)
#             cv.GetPad(1).SetMargin(0.135, 0.04, 0.45, 0.06)
#             cv.GetPad(2).SetMargin(0.135, 0.04, 0.15, 0.56)
#             cv.GetPad(1).SetLogy(1)
#             cv.cd(1)
            
#             statictics = ""
#             if daHists[logctau][idis][0].Integral()>0.:
#                 KS = daHists[logctau][idis][0].KolmogorovTest(daHists[logctau][idis][1], "M")
#                 daHists[logctau][idis][0].Scale(daHists[logctau][idis][1].Integral()/daHists[logctau][idis][0].Integral())
            
#                 eventsAboveWp = {
#                     50: [0.,0.],
#                     80: [0.,0.],
#                     95: [0.,0.],
#                 }
#                 sumMC = 0.
#                 for ibin in range(daHists[logctau][idis][0].GetNbinsX()):
#                     cMC = daHists[logctau][idis][0].GetBinContent(ibin+1)
#                     sumMC += cMC
#                     cData = daHists[logctau][idis][1].GetBinContent(ibin+1)

#                     for wp in eventsAboveWp.keys():
#                         if (sumMC/daHists[logctau][idis][0].Integral())>(wp*0.01):
#                             eventsAboveWp[wp][0]+=cMC
#                             eventsAboveWp[wp][1]+=cData
                
#                 for iwp,wp in enumerate(sorted(eventsAboveWp.keys())):
#                     statictics+="#Delta#epsilon#scale[0.7]{#lower[0.7]{%i}}: %+.1f%%"%(
#                         wp,
#                         100.*eventsAboveWp[wp][0]/daHists[logctau][idis][0].Integral()-100.*eventsAboveWp[wp][1]/daHists[logctau][idis][1].Integral()
#                     )
#                     if iwp<(len(eventsAboveWp.keys())-1):
#                         statictics+=","
#                     statictics+="  "
#                 statictics+="D = %.7f" % KS
#                 if idis == 4:
#                     KS_scores.append(KS)

#             daHists[logctau][idis][0].Rebin(200)
#             daHists[logctau][idis][1].Rebin(200)
#             ymax = max([daHists[logctau][idis][0].GetMaximum(),daHists[logctau][idis][1].GetMaximum()])
#             ymin = ymax
#             for ibin in range(daHists[logctau][idis][0].GetNbinsX()):
#                 cMC = daHists[logctau][idis][0].GetBinContent(ibin+1)
#                 cData = daHists[logctau][idis][1].GetBinContent(ibin+1)
#                 if cMC>1 and cData>1:
#                     ymin = min([ymin,cMC,cData])
                    
#             ymin = math.exp(math.log(ymax)-1.2*(math.log(ymax)-math.log(ymin)))
#             axis = ROOT.TH2F("axis"+str(idis)+str(random.random()),";;Resampled jets",50,0,1,50,ymin,math.exp(1.1*math.log(ymax)))
#             axis.GetXaxis().SetLabelSize(0)
#             axis.GetXaxis().SetTickLength(0.015/(1-cv.GetPad(1).GetLeftMargin()-cv.GetPad(1).GetRightMargin()))
#             axis.GetYaxis().SetTickLength(0.015/(1-cv.GetPad(1).GetTopMargin()-cv.GetPad(1).GetBottomMargin()))
#             axis.Draw("AXIS")
#             daHists[logctau][idis][0].Draw("HISTSAME")
#             daHists[logctau][idis][1].Draw("PESAME")
            
#             pText = ROOT.TPaveText(0.96,0.97,0.96,0.97,"NDC")
#             pText.SetTextFont(43)
#             pText.SetTextAlign(32)
#             pText.SetTextSize(30)
#             pText.AddText(statictics)
#             pText.Draw("Same")
            
#             cv.cd(2)
#             axisRes = ROOT.TH2F("axis"+str(idis)+str(random.random()),";Prob("+labels[idis]+", "+ctauSymbol(logctau=logctau)[0]+");Data/Pred.",50,0,1,50,0.2,1.8)
#             axisRes.Draw("AXIS")
#             axisRes.GetXaxis().SetTickLength(0.015/(1-cv.GetPad(2).GetLeftMargin()-cv.GetPad(2).GetRightMargin()))
#             axisRes.GetYaxis().SetTickLength(0.015/(1-cv.GetPad(2).GetTopMargin()-cv.GetPad(2).GetBottomMargin()))
#             axisLine = ROOT.TF1("axisLine","1",0,1)
#             axisLine.SetLineColor(ROOT.kBlack)
#             axisLine.Draw("Same")
#             axisLineUp = ROOT.TF1("axisLineUp","1.5",0,1)
#             axisLineUp.SetLineColor(ROOT.kBlack)
#             axisLineUp.SetLineStyle(2)
#             axisLineUp.Draw("Same")
#             axisLineDown = ROOT.TF1("axisLineDown","0.5",0,1)
#             axisLineDown.SetLineColor(ROOT.kBlack)
#             axisLineDown.SetLineStyle(2)
#             axisLineDown.Draw("Same")
#             daHistsRes = daHists[logctau][idis][1].Clone(daHists[logctau][idis][1].GetName()+"res")
#             daHistsRes.Divide(daHists[logctau][idis][0])
#             daHistsRes.Draw("PESAME")
#             cv.Print(os.path.join(outputFolder,"epoch_" + str(epoch),"da_"+str(logctau)+labels[idis].replace("||","_")+".pdf"))
#             cv.Print(os.path.join(outputFolder,"epoch_" + str(epoch),"da_"+str(logctau)+labels[idis].replace("||","_")+".png"))
    

# f = open(os.path.join(outputFolder, "model_epoch.stat"), "a")
# tight_WP_eff = np.mean(np.asarray(tight_WP_effs))
# KS_scores = np.asarray(KS_scores)
# f.write(str(epoch)+";"+str(learning_rate_val)+";"+str(avgLoss_train)+";"+str(avgLoss_test)+";"+str(avgLoss_train_domain)+";"+str(avgLoss_test_domain)+";"+str(classLossWeight)+";"+str(domainLossWeight)+";"+str(tight_WP_eff)+";"+np.array2string(KS_scores,separator=';')+"\n")
# f.close()

# cv = ROOT.TCanvas("cv"+str(idis)+str(random.random()),"",800,750)

# cv.Divide(1,3,0,0)
# cv.GetPad(1).SetPad(0.0, 0.0, 1.0, 1.0)
# cv.GetPad(2).SetPad(0.0, 0.0, 1.0, 1.0)
# cv.GetPad(3).SetPad(0.0, 0.0, 1.0, 1.0)
# cv.GetPad(1).SetFillStyle(4000)
# cv.GetPad(2).SetFillStyle(4000)
# cv.GetPad(3).SetFillStyle(4000)
# cv.GetPad(1).SetMargin(0.135, 0.04, 0.6, 0.06)
# cv.GetPad(2).SetMargin(0.135, 0.04, 0.27, 0.42)
# cv.GetPad(3).SetMargin(0.135, 0.04, 0.15, 0.75)
# #cv.GetPad(1).SetLogy(1)
# cv.GetPad(2).SetLogy(1)
# cv.GetPad(3).SetLogy(1)
# cv.cd(1)

# cv.SetMargin(0.135, 0.04, 0.13, 0.04)
# epocharray = np.linspace(1,len(lr_per_epoch),len(lr_per_epoch))
# axis1 = ROOT.TH2F("axis1"+str(random.random()),";Epoch;Loss",
#     50,0,len(lr_per_epoch)+1,
#     50,
#     0.85*min(avgLoss_train_per_epoch+avgLoss_test_per_epoch+avgLoss_train_domain_per_epoch+avgLoss_test_domain_per_epoch),
#     1.15*max(avgLoss_train_per_epoch+avgLoss_test_per_epoch+avgLoss_train_domain_per_epoch+avgLoss_test_domain_per_epoch)
# )
# axis1.GetXaxis().SetTickLength(0.015/(1-cv.GetLeftMargin()-cv.GetRightMargin()))
# axis1.GetYaxis().SetTickLength(0.015/(1-cv.GetTopMargin()-cv.GetBottomMargin()))
# axis1.Draw("AXIS")

# g_train_class = ROOT.TGraph(len(epocharray),epocharray,np.array(avgLoss_train_per_epoch))
# g_train_class.SetLineWidth(2)
# g_train_class.SetLineColor(ROOT.kAzure-4)
# g_train_class.SetMarkerColor(ROOT.kAzure-4)
# g_train_class.SetMarkerSize(1.2)
# g_train_class.Draw("PL")
# g_test_class = ROOT.TGraph(len(epocharray),epocharray,np.array(avgLoss_test_per_epoch))
# g_test_class.SetLineWidth(4)
# g_test_class.SetLineStyle(2)
# g_test_class.SetLineColor(ROOT.kBlue)
# g_test_class.Draw("L")
# g_train_domain = ROOT.TGraph(len(epocharray),epocharray,np.array(avgLoss_train_domain_per_epoch))
# g_train_domain.SetLineWidth(2)
# g_train_domain.SetLineColor(ROOT.kOrange+7)
# g_train_domain.SetMarkerColor(ROOT.kOrange+7)
# g_train_domain.SetMarkerSize(1.2)
# g_train_domain.Draw("PL")
# g_test_domain = ROOT.TGraph(len(epocharray),epocharray,np.array(avgLoss_test_domain_per_epoch))
# g_test_domain.SetLineWidth(4)
# g_test_domain.SetLineStyle(2)
# g_test_domain.SetLineColor(ROOT.kRed+1)
# g_test_domain.Draw("L")
# cv.Print(os.path.join(outputFolder,"epoch_" + str(epoch),"loss.pdf"))

# lr_per_epoch = []
# class_weight_per_epoch = []
# domain_weight_per_epoch = []
    
# avgLoss_train_per_epoch = []
# avgLoss_test_per_epoch = []
# avgLoss_train_domain_per_epoch = []
# avgLoss_test_domain_per_epoch = []
# if epoch > 1 and previous_train_loss < avgLoss_train:
#     learning_rate_val = learning_rate_val*0.85
#     print "Decreasing learning rate to %.4e" % (learning_rate_val)

# previous_train_loss = avgLoss_train

# coord.request_stop()
# coord.join(threads)
# K.clear_session()
# epoch += 1
