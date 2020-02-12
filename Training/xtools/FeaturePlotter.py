import ROOT
import random
import numpy as np
import math
import os
import sys
import logging
import signal

class FeaturePlotter():
    def __init__(self, featureDict, maxEntries = -1):
        self.featureDict = featureDict
        self.maxEntries = maxEntries
        self.data = {}
        for featureGroupName,featureGroup in self.featureDict.items():
            self.data[featureGroupName] = []
            
            branchNames = featureGroup['names'] if featureGroup.has_key('names') else featureGroup['branches']

            for branchName in branchNames:
                self.data[featureGroupName].append({"name":branchName, "array":[]})
                
    def fill(self,batch):
        
        for featureGroupName,featureGroup in self.featureDict.items():
            if self.maxEntries>0 and len(self.data[featureGroupName][0]['array'])>self.maxEntries:
                continue
            for ibranch in range(len(featureGroup['branches'])):
                for ibatch in range(batch[featureGroupName].shape[0]):
                    if featureGroup.has_key('max'):
                        #exclude zero-padded entries
                        notZeroPadded = np.amax(np.abs(batch[featureGroupName][ibatch]),axis=1)>1e-12
                        if np.any(notZeroPadded):
                            values = batch[featureGroupName][ibatch,:,ibranch][notZeroPadded]
                            if np.isnan(values).any():
                                logging.critical("NAN value found in inputs for '"+self.data[featureGroupName][ibranch]['name']+"' of group '"+featureGroupName+"'")
                                sys.exit(1)
                            if np.isinf(values).any():
                                logging.critical("INF value found in inputs for '"+self.data[featureGroupName][ibranch]['name']+"' of group '"+featureGroupName+"'")
                                sys.exit(1)
                            if np.any(np.abs(values)>1e12):
                                logging.critical("Large values ("+str(values)+") found in inputs for '"+self.data[featureGroupName][ibranch]['name']+"' of group '"+featureGroupName+"'")
                                sys.exit(1)
                            self.data[featureGroupName][ibranch]['array'].append(
                                np.mean(values)
                            )
                    else:
                        value = batch[featureGroupName][ibatch,ibranch]
                        if np.isnan(value):
                            logging.critical("NAN value found in inputs for '"+self.data[featureGroupName][ibranch]['name']+"' of group '"+featureGroupName+"'")
                            sys.exit(1)
                        if np.isinf(value):
                            logging.critical("INF value found in inputs for '"+self.data[featureGroupName][ibranch]['name']+"' of group '"+featureGroupName+"'")
                            sys.exit(1)
                        if np.abs(value)>1e12:
                            logging.critical("Large value ("+str(value)+") found in inputs for '"+self.data[featureGroupName][ibranch]['name']+"' of group '"+featureGroupName+"'")
                            sys.exit(1)
                        self.data[featureGroupName][ibranch]['array'].append(value)
                        
    def plot(self,path):
        for featureGroupName,featureGroup in self.featureDict.items():
            for ibranch in range(len(featureGroup['branches'])):
                cv = ROOT.TCanvas("cv"+featureGroupName+str(ibranch)+str(random.random()),"",800,670)
                cv.SetMargin(0.13,0.09,0.15,0.07)
                #cv.SetLogy(1)
                xmin = min(self.data[featureGroupName][ibranch]['array'])
                xmax = max(self.data[featureGroupName][ibranch]['array'])
                hist = ROOT.TH1F(
                    "hist"+featureGroupName+str(ibranch)+str(random.random()),
                    ";"+self.data[featureGroupName][ibranch]['name']+";Entries",
                    int(1+round(0.1*math.sqrt(len(self.data[featureGroupName][ibranch]['array'])))),
                    xmin-0.05*(xmax-xmin),
                    xmax+0.05*(xmax-xmin),
                )
                for value in self.data[featureGroupName][ibranch]['array']:
                    hist.Fill(value)
                hist.Draw("HIST")
                hist.GetYaxis().SetRangeUser(0.7,10**(1.1*math.log10(max(1.,hist.GetMaximum()))))
                
                cv.Print(os.path.join(path,featureGroupName+"_"+self.data[featureGroupName][ibranch]['name']+".pdf"))
                
