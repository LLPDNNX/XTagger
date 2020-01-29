import ROOT
import logging
import sys
import random
import copy
import numpy as np
import xtools

class Weights():
    def __init__(
        self,
        labelNameList,
        weightHistsPerLabel,
        ptBinning,
        etaBinning
    ):
        self.labelNameList = labelNameList
        self.weightHistsPerLabel = weightHistsPerLabel
        self.ptBinning = ptBinning
        self.etaBinning = etaBinning
        
    def plot(self,path):
        cv = ROOT.TCanvas("cv"+str(random.random()),"",800,700)
        cv.Divide(1,2)
        cv.GetPad(1).SetMargin(0.13,0.03,0.17,0.1)
        cv.GetPad(2).SetMargin(0.13,0.03,0.17,0.1)
        
        color=[ROOT.kAzure-4,ROOT.kOrange+7,ROOT.kGreen+1,ROOT.kBlue+1,ROOT.kGray,ROOT.kGray+1,ROOT.kGray+2,ROOT.kBlack]
        
        ptNumeratorHists = {k: v['numerator'].ProjectionX() for k,v in self.weightHistsPerLabel.items()}
        ptDenominatorHists = {k: v['denominator'].ProjectionX() for k,v in self.weightHistsPerLabel.items()}
        ptWeightHists = {}
        
        etaNumeratorHists = {k: v['numerator'].ProjectionY() for k,v in self.weightHistsPerLabel.items()}
        etaDenominatorHists = {k: v['denominator'].ProjectionY() for k,v in self.weightHistsPerLabel.items()}
        etaWeightHists = {}
        
        for k in self.labelNameList:
            ptWeightHists[k] = ptNumeratorHists[k].Clone()
            ptWeightHists[k].Divide(ptDenominatorHists[k])
            ptWeightHists[k].SetDirectory(0)
            
            etaWeightHists[k] = etaNumeratorHists[k].Clone()
            etaWeightHists[k].Divide(etaDenominatorHists[k])
            etaWeightHists[k].SetDirectory(0)
            
        cv.cd(1)
            
        axisPt = ROOT.TH2F("axisPt"+str(random.random()),";Jet p#lower[0.2]{#scale[0.8]{T}} (GeV);Weight",
            len(self.ptBinning)-1,self.ptBinning,
            50,np.linspace(0,1.1*max(map(lambda x: x.GetMaximum(),ptWeightHists.values())),51)
        )
        axisPt.Draw("AXIS")
        
        for i,k in enumerate(self.labelNameList):
            ptWeightHists[k].SetLineColor(color[i])
            ptWeightHists[k].SetLineWidth(2)
            ptWeightHists[k].Draw("SameHIST")
            
        cv.cd(2)
            
        axisEta = ROOT.TH2F("axisEta"+str(random.random()),";Jet #eta;Weight",
            len(self.etaBinning)-1,self.etaBinning,
            50,np.linspace(0,1.1*max(map(lambda x: x.GetMaximum(),etaWeightHists.values())),51)
        )
        axisEta.Draw("AXIS")
        
        for i,k in enumerate(self.labelNameList):
            etaWeightHists[k].SetLineColor(color[i])
            etaWeightHists[k].SetLineWidth(2)
            etaWeightHists[k].Draw("SameHIST")
            
        cv.Print(path)
        
    def save(self,path):
        outputFile = ROOT.TFile(path,"RECREATE")
        for k in self.labelNameList:
            hist = self.weightHistsPerLabel[k]['weight'].Clone(k)
            hist.SetDirectory(outputFile)
            hist.Write()
        outputFile.Close()
        
        
class ResampledDistribution():
    def __init__(
        self,
        labelNameList,
        ptBinning,
        etaBinning,
        paramBinning,
    ):
        self.labelNameList = labelNameList
        self.ptBinning = ptBinning
        self.etaBinning = etaBinning
        self.paramBinning = paramBinning
        self.ptHists = []
        self.etaHists = []
        self.paramHists = []
        for k in self.labelNameList:
            self.ptHists.append(ROOT.TH1F(k+"pt"+str(random.random()),";Jet p#lower[0.2]{#scale[0.8]{T}} (GeV); #Jets",len(self.ptBinning)-1,self.ptBinning))
            self.etaHists.append(ROOT.TH1F(k+"eta"+str(random.random()),";Jet #eta; #Jets",len(self.etaBinning)-1,self.etaBinning))
            self.paramHists.append(ROOT.TH1F(k+"param"+str(random.random()),";Jet displacement; #Jets",len(self.paramBinning)-1,self.paramBinning))
            
        for i,k in enumerate(self.labelNameList):
            self.ptHists[i].SetDirectory(0)
            self.etaHists[i].SetDirectory(0)
            self.paramHists[i].SetDirectory(0)
            
    def fill(self,truthValue,ptValues,etaValues,paramValues):
        labelIndices = np.argmax(truthValue,axis=1)
        for i in range(labelIndices.shape[0]):
            self.ptHists[labelIndices[i]].Fill(ptValues[i])
            self.etaHists[labelIndices[i]].Fill(etaValues[i])
            self.paramHists[labelIndices[i]].Fill(paramValues[i])
            
    def plot(self,path):
        cv = ROOT.TCanvas("cv"+str(random.random()),"",800,900)
        cv.Divide(1,3)
        cv.GetPad(1).SetMargin(0.13,0.03,0.17,0.1)
        cv.GetPad(2).SetMargin(0.13,0.03,0.17,0.1)
        cv.GetPad(3).SetMargin(0.13,0.03,0.17,0.1)
        
        color=[ROOT.kAzure-4,ROOT.kOrange+7,ROOT.kGreen+1,ROOT.kBlue+1,ROOT.kGray,ROOT.kGray+1,ROOT.kGray+2,ROOT.kBlack]
        
        cv.cd(1)
        axisPt = ROOT.TH2F(
            "axispt"+str(random.random()),";Jet p#lower[0.2]{#scale[0.8]{T}} (GeV); #Jets",
            len(self.ptBinning)-1,self.ptBinning,
            50,np.linspace(0,1.1*max(map(lambda x: x.GetMaximum(),self.ptHists)),51)
        )
        axisPt.Draw("AXIS")

        for i,k in enumerate(self.labelNameList):
            self.ptHists[i].SetLineColor(color[i])
            self.ptHists[i].SetLineWidth(2)
            self.ptHists[i].Draw("SameHIST")
        
        cv.cd(2)
        axisEta = ROOT.TH2F(
            "axiseta"+str(random.random()),";Jet #eta; #Jets",
            len(self.etaBinning)-1,self.etaBinning,
            50,np.linspace(0,1.1*max(map(lambda x: x.GetMaximum(),self.etaHists)),51)
        )
        axisEta.Draw("AXIS")
        for i,k in enumerate(self.labelNameList):
            self.etaHists[i].SetLineColor(color[i])
            self.etaHists[i].SetLineWidth(2)
            self.etaHists[i].Draw("SameHIST")
        
        cv.cd(3)
        axisParam = ROOT.TH2F(
            "axisparam"+str(random.random()),";Jet displacement; #Jets",
            len(self.paramBinning)-1,self.paramBinning,
            50,np.linspace(0,1.1*max(map(lambda x: x.GetMaximum(),self.paramHists)),51)
        )
        axisParam.Draw("AXIS")
        for i,k in enumerate(self.labelNameList):
            self.paramHists[i].SetLineColor(color[i])
            self.paramHists[i].SetLineWidth(2)
            self.paramHists[i].Draw("SameHIST")
        
        cv.Print(path)

class ResampleWeights():
    def __init__(
        self,
        fileList,
        labelNameList,
        labelWeightList,
        targetWeight,
        ptBinning=np.logspace(1,2,20),
        etaBinning=np.linspace(-2.4,2.4,10)
    ):
        self.labelNameList = labelNameList
        self.labelWeightList = labelWeightList
        self.targetWeight = targetWeight
        self.ptBinning = ptBinning
        self.etaBinning = etaBinning
        
        self.labelNameList = labelNameList
        self.histPerLabel = {}
        
        if len(labelNameList)!=len(labelWeightList):
            logging.critical("Length of label names and weights do not match!"%(len(labelNameList),len(labelWeightList)))
            sys.exit(1)
        
        chain = ROOT.TChain("jets")
        for f in fileList:
            chain.AddFile(f)
        nEntries = chain.GetEntries()
        if nEntries<100:
            logging.critical("Not enough entries: %i"%nEntries)
            sys.exit(1)
        logging.info("Total entries: %i"%nEntries)
        
        for ilabel,labelWeight in enumerate(self.labelWeightList):
            hist = self.projectHist(chain,self.labelNameList[ilabel],labelWeight)
            self.histPerLabel[self.labelNameList[ilabel]] = hist
            
        self.histTarget = self.projectHist(chain,'target',self.targetWeight)
        self.totalIntegral = sum(map(lambda x: x.Integral(),self.histPerLabel.values()))
        if self.totalIntegral<100:
            logging.critical("Total number of jets too small:"+str(self.totalIntegral))
            sys.exit(1)
            
        msg = "Class balance before resampling: "
        for ilabel,labelWeight in enumerate(self.labelWeightList):
            msg += "%s: %.1f%%; "%(
                labelNameList[ilabel],
                100.*self.histPerLabel[self.labelNameList[ilabel]].Integral()/self.totalIntegral
            )
        msg += "target: %.1f%%"%(
            100.*self.histTarget.Integral()/self.totalIntegral
        )
        logging.info(msg)
        
    def reweight(self,classBalance=True,threshold=10,oversampling=2.):
        weightHistsPerLabel = {}
        maxWeightPerLabel = {}
        for k in self.histPerLabel.keys():
            weightHistsPerLabel[k] = {}
            maxWeightPerLabel[k] = 0
            
        for ibin in range(self.histTarget.GetNbinsX()):
            for jbin in range(self.histTarget.GetNbinsY()):
                nTarget = self.histTarget.GetBinContent(ibin+1,jbin+1)
                if nTarget<threshold:
                    continue
                for k in self.histPerLabel.keys():
                    nLabel = self.histPerLabel[k].GetBinContent(ibin+1,jbin+1)
                    if nLabel<threshold:
                        continue
                    maxWeightPerLabel[k] = max(maxWeightPerLabel[k],nTarget/nLabel)
        maxWeight = max(maxWeightPerLabel.values())/oversampling
        
        for k in self.histPerLabel.keys():
            weightHist = ROOT.TH2F(
                k+"weight"+str(random.random()),"",
                len(self.ptBinning)-1, self.ptBinning,
                len(self.etaBinning)-1, self.etaBinning
            )
            weightHist.SetDirectory(0)
            
            numeratorHist = ROOT.TH2F(
                k+"numerator"+str(random.random()),"",
                len(self.ptBinning)-1, self.ptBinning,
                len(self.etaBinning)-1, self.etaBinning
            )
            numeratorHist.SetDirectory(0)
            
            denominatorHist = ROOT.TH2F(
                k+"denominator"+str(random.random()),"",
                len(self.ptBinning)-1, self.ptBinning,
                len(self.etaBinning)-1, self.etaBinning
            )
            denominatorHist.SetDirectory(0)
            
            for ibin in range(self.histTarget.GetNbinsX()):
                for jbin in range(self.histTarget.GetNbinsY()):
                    nTarget = self.histTarget.GetBinContent(ibin+1,jbin+1)/maxWeight
                    nLabel = self.histPerLabel[k].GetBinContent(ibin+1,jbin+1)
                    if nTarget>threshold and nLabel>threshold:
                        weightHist.SetBinContent(ibin+1,jbin+1,nTarget/nLabel)
                        numeratorHist.SetBinContent(ibin+1,jbin+1,nTarget)
                        denominatorHist.SetBinContent(ibin+1,jbin+1,nLabel)
                    else:
                        weightHist.SetBinContent(ibin+1,jbin+1,0)
                        numeratorHist.SetBinContent(ibin+1,jbin+1,0)
                        denominatorHist.SetBinContent(ibin+1,jbin+1,0)
                    
            weightHistsPerLabel[k]['weight'] = weightHist
            weightHistsPerLabel[k]['numerator'] = numeratorHist
            weightHistsPerLabel[k]['denominator'] = denominatorHist
        return Weights(self.labelNameList, weightHistsPerLabel,self.ptBinning,self.etaBinning)
        
    def projectHist(self,chain,histName,labelWeight):
        hist = ROOT.TH2F(
            histName+str(random.random()), "", 
            len(self.ptBinning)-1, self.ptBinning,
            len(self.etaBinning)-1, self.etaBinning
        )
        hist.Sumw2()
        chain.Project(
            hist.GetName(),
            "global_eta:global_pt",
            "("+labelWeight+">0.5)"
        )
                      
        hist.SetDirectory(0)
        if hist.Integral() > 0:
            logging.info( "Entries for label '%s' and weight '%s' = %i"%(histName,labelWeight,hist.Integral()))
        else:
            logging.warning("No entries found for label '%s' and weight '%s'"%(histName,labelWeight))
        return hist
        
    def makeDistribution(self,paramBinning):
        return ResampledDistribution(
            self.labelNameList,
            self.ptBinning,
            self.etaBinning,
            paramBinning
        )
        
    def getLabelNameList(self):
        return self.labelNameList
        
    def plot(self,path):
        cv = xtools.style.makeCanvas(name="cv"+str(random.random()))
        ptHists = {k: v.ProjectionX() for k,v in self.histPerLabel.items()}
        axisPt = ROOT.TH2F("axisPt"+str(random.random()),";pT bin;Weight",
            len(self.ptBinning)-1,self.ptBinning,
            50,np.linspace(0,1.1*max(map(lambda x: x.GetMaximum(),ptHists.values())),51)
        )
        axisPt.Draw("AXIS")
        color=[ROOT.kAzure-4,ROOT.kOrange+7,ROOT.kGreen+1,ROOT.kBlue+1,ROOT.kGray,ROOT.kGray+1,ROOT.kGray+2,ROOT.kBlack]
        for i,k in enumerate(self.labelNameList):
            ptHists[k].SetLineColor(color[i])
            ptHists[k].SetLineWidth(2)
            ptHists[k].Draw("SameHIST")
        cv.Print(path)
            
        

