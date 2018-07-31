import ROOT
import math
import numpy as np
import os
import random
from sklearn.metrics import auc
from root_style import set_root_style
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score

NRGBs = 6;
NCont = 255;
cvscale = 1.
fontScale = 750./650.

stops = np.array( [0.00, 0.34, 0.47, 0.61, 0.84, 1.00] )
red  = np.array( [0.5, 0.00, 0.1, 1., 1.00, 0.81] )
green = np.array( [0.10, 0.71, 0.85, 0.70, 0.20, 0.00] )
blue = np.array( [0.91, 1.00, 0.12, 0.1, 0.00, 0.00] )

colWheelDark = ROOT.TColor.CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont)

for i in range(NRGBs):
    red[i]=min(1,red[i]*1.1+0.25)
    green[i]=min(1,green[i]*1.1+0.25)
    blue[i]=min(1,blue[i]*1.1+0.25)

colWheel = ROOT.TColor.CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont)
def find_nearest(sigArray, bgArray, value):
    sigArray = np.asarray(sigArray)
    bgArray = np.asarray(bgArray)
    idx = (np.abs(bgArray - value)).argmin()
    return sigArray[idx], bgArray[idx]

colors=[]
def hex2rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16)/255.0 for i in range(0, lv, lv // 3))

def newColor(red,green,blue):
    newColor.colorindex+=1
    color=ROOT.TColor(newColor.colorindex,red,green,blue)
    colors.append(color)
    return color

newColor.colorindex=301
def getDarkerColor(color):
    darkerColor=newColor(color.GetRed()*0.6,color.GetGreen()*0.6,color.GetBlue()*0.6)
    return darkerColor

style = {
    'isB':[newColor(0.8,0.45,0),3,1],
    'isBB':[newColor(0.85,0.42,0),3,2],
    'isGBB':[newColor(0.9,0.39,0),2,1],
    'isLeptonicB':[newColor(0.95,0.36,0),3,2],
    'isLeptonicB_C':[newColor(1,0.33,0),2,1],

    'isC':[newColor(0,0.9,0.1),3,2],
    'isCC':[newColor(0,0.8,0.25),2,1],
    'isGCC':[newColor(0,0.7,0.35),3,2],

    'isUD':[newColor(0.65,0.65,0.65),3,1],
    'isS':[newColor(0.55,0.55,0.55),3,2],
    'isG':[newColor(0.45,0.45,0.45),3,1],
    'isUndefined':[newColor(0.4,0.4,0.4),3,2],

    'isFromLLgno_isB':[newColor(0.0,0.1,1),3,1],
    'isFromLLgno_isBB':[newColor(0.0,0.13,0.95),3,2],
    'isFromLLgno_isGBB':[newColor(0.0,0.16,0.9),2,1],
    'isFromLLgno_isLeptonicB':[newColor(0.0,0.19,0.87),3,1],
    'isFromLLgno_isLeptonicB_C':[newColor(0.0,0.22,0.85),3,2],
    'isFromLLgno_isC':[newColor(0.0,0.25,0.83),2,1],
    'isFromLLgno_isCC':[newColor(0.0,0.28,0.8),3,2],
    'isFromLLgno_isGCC':[newColor(0.0,0.31,0.77),2,1],
    'isFromLLgno_isUD':[newColor(0.0,0.34,0.75),3,2],
    'isFromLLgno_isS':[newColor(0.0,0.37,0.73),2,1],
    'isFromLLgno_isG':[newColor(0.0,0.4,0.7),3,2],
    'isFromLLgno_isUndefined':[newColor(0.0,0.43,0.67),2,1],
}

# Draw hist by type: ll, b, c or usdg
def drawHists(histDict,branchNameList,legend):

    ll = 4
    b = 4
    c = 4
    other = 4
    for label in branchNameList:
        hist = histDict[label]
        legend.AddEntry(hist,label.replace("is",""),"L")
        if label.find("fromLLP")>=0:
            hist.SetLineColor(ROOT.kOrange+7)
            hist.SetLineWidth(ll/3)
            hist.SetLineStyle(ll%3+1)
            ll+=1
        elif label.find("B")>0:
            hist.SetLineColor(ROOT.kAzure-4)
            hist.SetLineWidth(b/2)
            hist.SetLineStyle(b%2+1)
            b+=1
        elif label.find("C")>0:
            hist.SetLineColor(ROOT.kGreen)
            hist.SetLineWidth(c/2)
            hist.SetLineStyle(c%2+1)
            c+=1
        else:
            hist.SetLineColor(ROOT.kMagenta)
            hist.SetLineWidth(other/2)
            hist.SetLineStyle(other%2+1)
            other+=1
        hist.Draw("SameHISTL")

# Make a plot: need outputfolder, dictionary of hists, list of branch names and 
# hist to normalise with respect to 
def makePlot(outputFolder, histDict, branchNameList, binning, title, output, 
      target=None, logx=0, logy=0):
    cv = ROOT.TCanvas("cv","",1100,700)
    cv.SetLogx(logx)
    cv.SetLogy(logy)
    cv.SetRightMargin(0.36)

    ymax = max(map(lambda h: h.GetMaximum(),histDict.values()))
    axis = ROOT.TH2F("axis", title, 50, binning[0], binning[-1], 50, 0, ymax*1.1)
    axis.Draw("AXIS")

    legend = ROOT.TLegend(0.67,0.98,0.99,0.02)
    legend.SetBorderSize(0)
    legend.SetFillColor(ROOT.kWhite)
    legend.SetTextFont(43)
    legend.SetTextSize(22*cvscale*fontScale)
    drawHists(histDict,branchNameList,legend)

    if target:
        target.SetLineWidth(3)
        target.SetLineColor(ROOT.kBlack)
        target.Draw("SameHISTL")
        legend.AddEntry(target,"Target","L")
    legend.Draw("Same")

    cv.Update()
    cv.Print(os.path.join(outputFolder,output+".pdf"))
    cv.Print(os.path.join(outputFolder,output+".root"))

# Get ROC curve for given signal and background distributions
def getROC(signal,background):

    # number of bins
    N = signal.GetNbinsX()+2
    sigHistContent = np.zeros(N)
    bgHistContent = np.zeros(N)

    for i in range(N):
        sigHistContent[i] = signal.GetBinContent(i)
        bgHistContent[i] = background.GetBinContent(i)

    sigN = sum(sigHistContent)
    bgN = sum(bgHistContent)

    sigEff = []
    bgRej = []
    bgEff = []
    sig_integral = 0.0
    bg_integral = 0.0

    for ibin in reversed(range(N)):
        sig_integral += sigHistContent[ibin]
        bg_integral += bgHistContent[ibin]
        sigEff.append(sig_integral / sigN)
        bgRej.append(1 - bg_integral/bgN)
        bgEff.append(bg_integral / bgN)
    return sigEff, bgRej, bgEff

def getAUC(sigEff, bgRej):
    integral=0.0
    for i in range(len(sigEff) - 1):
        w = math.fabs(sigEff[i+1] - sigEff[i])
        h = 0.5*(bgRej[i+1] + bgRej[i])
        x = (sigEff[i+1] + sigEff[i])*0.5
        integral +=w * math.fabs(h - (1-x))
    return math.fabs(integral)

def drawROC(name,sigEff,bgEff,signalName="Signal",backgroundName="Background",auc=None,style=1):

    cv = ROOT.TCanvas("cv_roc"+str(random.random()),"",800,600)
    cv.SetPad(0.0, 0.0, 1.0, 1.0)
    cv.SetFillStyle(4000)

    cv.SetBorderMode(0)
    #cv.SetGridx(True)
    #cv.SetGridy(True)

    #For the frame:
    cv.SetFrameBorderMode(0)
    cv.SetFrameBorderSize(1)
    cv.SetFrameFillColor(0)
    cv.SetFrameFillStyle(0)
    cv.SetFrameLineColor(1)
    cv.SetFrameLineStyle(1)
    cv.SetFrameLineWidth(1)

    # Margins:
    cv.SetLeftMargin(0.163)
    cv.SetRightMargin(0.03)
    cv.SetTopMargin(0.08)
    cv.SetBottomMargin(0.175)

    # For the Global title:
    cv.SetTitle("")

    # For the axis:
    cv.SetTickx(1)  # To get tick marks on the opposite side of the frame
    cv.SetTicky(1)

    cv.SetLogy(1)

    axis=ROOT.TH2F("axis" + str(random.random()),";" + signalName + " efficiency;" + backgroundName + " efficiency", 50, 0, 1.0, 50, 0.0008, 1.0)
    axis.GetYaxis().SetNdivisions(508)
    axis.GetXaxis().SetNdivisions(508)
    axis.GetXaxis().SetTickLength(0.015/(1-cv.GetLeftMargin()-cv.GetRightMargin()))
    axis.GetYaxis().SetTickLength(0.015/(1-cv.GetTopMargin()-cv.GetBottomMargin()))
    #axis.GetYaxis().SetNoExponent(True)
    axis.Draw("AXIS")

    #### draw here
    graphF = ROOT.TGraph(len(sigEff),np.array(sigEff),np.array(bgEff))
    graphF.SetLineWidth(0)
    graphF.SetFillColor(ROOT.kOrange+10)
    #graphF.Draw("SameF")

    graphL = ROOT.TGraph(len(sigEff),np.array(sigEff),np.array(bgEff))
    graphL.SetLineColor(ROOT.kOrange+7)
    graphL.SetLineWidth(3)
    graphL.SetLineStyle(style)
    graphL.Draw("SameL")

    ROOT.gPad.RedrawAxis()

    pCMS=ROOT.TPaveText(cv.GetLeftMargin(),0.94,cv.GetLeftMargin(),0.94,"NDC")
    pCMS.SetFillColor(ROOT.kWhite)
    pCMS.SetBorderSize(0)
    pCMS.SetTextFont(63)
    pCMS.SetTextSize(30*cvscale*fontScale)
    pCMS.SetTextAlign(11)
    pCMS.AddText("CMS")
    pCMS.Draw("Same")

    pPreliminary=ROOT.TPaveText(cv.GetLeftMargin()+0.095,0.94,cv.GetLeftMargin()+0.095,0.94,"NDC")
    pPreliminary.SetFillColor(ROOT.kWhite)
    pPreliminary.SetBorderSize(0)
    pPreliminary.SetTextFont(53)
    pPreliminary.SetTextSize(30*cvscale*fontScale)
    pPreliminary.SetTextAlign(11)
    pPreliminary.AddText("Simulation")
    pPreliminary.Draw("Same")

    if auc:
        pAUC=ROOT.TPaveText(1-cv.GetRightMargin(),0.94,1-cv.GetRightMargin(),0.94,"NDC")
        pAUC.SetFillColor(ROOT.kWhite)
        pAUC.SetBorderSize(0)
        pAUC.SetTextFont(43)
        pAUC.SetTextSize(32*cvscale*fontScale)
        pAUC.SetTextAlign(31)
        pAUC.AddText("AUC: % 4.1f %%" % (auc*100.0))
        pAUC.Draw("Same")

    cv.Update()
    cv.Print(name+".pdf")
    cv.Print(name+".root")

# fn for plotting arrays produced when training
def plot_resampled(outputFolder, x, xlabel, var_array, var_binning, truth_array):

        var_0 = var_array[truth_array == 0]
        var_1 = var_array[truth_array == 1]
        var_2 = var_array[truth_array == 2]
        var_3 = var_array[truth_array == 3]
        var_4 = var_array[truth_array == 4]

        fig = plt.figure()
        plt.hist([var_0, var_1, var_2, var_3, var_4], 
                bins=var_binning, label=['b', 'c', 'uds', 'g', 'llp'], alpha=0.5)
        plt.legend(loc='upper right')
        plt.xlabel(xlabel)
        plt.ylabel("# entries/ bin")
        plt.savefig(os.path.join(outputFolder, "reweighted_"+x+".pdf"))
        plt.close(fig)

def make_plots(outputFolder, epoch, hists, truths, scores, featureDict):
    set_root_style()

    epoch_path = os.path.join(outputFolder, "epoch_" + str(epoch))
    if not os.path.exists(epoch_path):
        os.makedirs(epoch_path)

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
        tight_wps_sig = []
        tight_wps_bg = []
        tight_lines_x = []
        tight_lines_y = []

        for prob_label in range(dimension):
            if truth_label==prob_label:
                continue
            bkgHist = hists[truth_label][prob_label]
            sigEff, bgRej, bgEff = getROC(signalHist,bkgHist)
            auc = getAUC(sigEff,bgRej) + 0.5
            length = len(sigEff)
            sigEff = np.array(sigEff)
            bgEff = np.array(bgEff)

            sig_loose, bg_loose = find_nearest(sigEff, bgEff, 1e-1)
            sig_medium, bg_medium = find_nearest(sigEff, bgEff, 1e-2)
            sig_tight, bg_tight = find_nearest(sigEff, bgEff, 1e-3)
            tight_wps_sig.append(sig_tight)
            tight_wps_bg.append(bg_tight)

            tight_line_x = ROOT.TLine(sig_tight, 0, sig_tight, bg_tight)
            tight_line_x.SetLineColor(int(colWheelDark+250.*prob_label/(len(featureDict["truth"]["branches"])-1)))
            tight_line_x.SetLineStyle(2)
            tight_line_x.SetLineWidth(2)
            tight_lines_x.append(tight_line_x)
            tight_line_y = ROOT.TLine(0, bg_tight, sig_tight, bg_tight)
            tight_line_y.SetLineColor(ROOT.kBlack)
            tight_line_y.SetLineWidth(2)
            tight_lines_y.append(tight_line_y)
     
            #print sig_loose, bg_loose
            #print sig_medium, bg_medium
            #print sig_tight, bg_tight

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
            tight_lines_x[prob_label].Draw("SAMEL")
            tight_lines_y[prob_label].Draw("SAMEL")
            legend.AddEntry(roc,name[prob_label],"L")
            legend.AddEntry("","tight WP eff: %.1f%%"%(tight_wps_sig[prob_label]*100.),"")
        legend.Draw("Same")
        
        cv.Print(os.path.join(outputFolder, "epoch_"+str(epoch), "roc_"+names[truth_label]+".pdf"))
        cv.Print(os.path.join(outputFolder, "epoch_"+str(epoch), "roc_"+names[truth_label]+".root"))

        for prob_label in range(truth_label):
            average_auc = .5*(all_aucs[prob_label,truth_label] + all_aucs[truth_label, prob_label])
            print "average auc between ", names[truth_label]," and ", names[prob_label]," is: ", average_auc
            M_score += average_auc
            
    M_score = 2.*M_score/((dimension)*(dimension-1)) 
    print "-"*100
    print "The M score is: ", M_score
    print "-"*100
    
    rootOutput = ROOT.TFile(os.path.join(outputFolder, "epoch_"+str(epoch),"report.root"),"RECREATE")
    
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
        cv.Print(os.path.join(outputFolder,"epoch_"+str(epoch),disName+".pdf"))
        cv.Print(os.path.join(outputFolder,"epoch_"+str(epoch),disName+".root"))
        
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
    cv.Print(os.path.join(outputFolder,"epoch_"+str(epoch),"confusion.pdf"))
    rootOutput.Write()
    rootOutput.Close()

    return M_score


