import ROOT
import math
import numpy as np
import os
from sklearn.metrics import auc

global cvscale, fontScale, colWheel, colWheelDark
cvscale = 1.
fontScale = 750./650.

def set_root_style():
        
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(0)
    ROOT.gROOT.SetStyle("Plain")
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(1111)
    ROOT.gStyle.SetPadTopMargin(0.08)
    ROOT.gStyle.SetPadLeftMargin(0.145)
    ROOT.gStyle.SetPadRightMargin(0.26)
    ROOT.gStyle.SetPadBottomMargin(0.15)
    ROOT.gStyle.SetStatFontSize(0.025)

    ROOT.gStyle.SetOptFit()
    ROOT.gStyle.SetOptStat(0)

    # For the canvas:
    ROOT.gStyle.SetCanvasBorderMode(0)
    ROOT.gStyle.SetCanvasColor(ROOT.kWhite)
    ROOT.gStyle.SetCanvasDefH(700) #Height of canvas
    ROOT.gStyle.SetCanvasDefW(800) #Width of canvas
    ROOT.gStyle.SetCanvasDefX(0)   #POsition on screen
    ROOT.gStyle.SetCanvasDefY(0)

    # For the Pad:
    ROOT.gStyle.SetPadBorderMode(0)
    # ROOT.gStyle.SetPadBorderSize(Width_t size = 1)
    ROOT.gStyle.SetPadColor(ROOT.kWhite)
    #ROOT.gStyle.SetPadGridX(True)
    #ROOT.gStyle.SetPadGridY(True)
    ROOT.gStyle.SetGridColor(ROOT.kBlack)
    ROOT.gStyle.SetGridStyle(2)
    ROOT.gStyle.SetGridWidth(1)

    # For the frame:

    ROOT.gStyle.SetFrameBorderMode(0)
    ROOT.gStyle.SetFrameBorderSize(0)
    ROOT.gStyle.SetFrameFillColor(0)
    ROOT.gStyle.SetFrameFillStyle(0)
    ROOT.gStyle.SetFrameLineColor(1)
    ROOT.gStyle.SetFrameLineStyle(1)
    ROOT.gStyle.SetFrameLineWidth(0)

    # For the histo:
    # ROOT.gStyle.SetHistFillColor(1)
    # ROOT.gStyle.SetHistFillStyle(0)
    # ROOT.gStyle.SetLegoInnerR(Float_t rad = 0.5)
    # ROOT.gStyle.SetNumberContours(Int_t number = 20)

    ROOT.gStyle.SetEndErrorSize(2)
    #ROOT.gStyle.SetErrorMarker(20)
    ROOT.gStyle.SetErrorX(0.)

    ROOT.gStyle.SetMarkerStyle(20)
    #ROOT.gStyle.SetMarkerStyle(20)

    #For the fit/function:
    ROOT.gStyle.SetOptFit(1)
    ROOT.gStyle.SetFitFormat("5.4g")
    ROOT.gStyle.SetFuncColor(2)
    ROOT.gStyle.SetFuncStyle(1)
    ROOT.gStyle.SetFuncWidth(1)

    #For the date:
    ROOT.gStyle.SetOptDate(0)
    # ROOT.gStyle.SetDateX(Float_t x = 0.01)
    # ROOT.gStyle.SetDateY(Float_t y = 0.01)

    # For the statistics box:
    ROOT.gStyle.SetOptFile(0)
    ROOT.gStyle.SetOptStat(0) # To display the mean and RMS:   SetOptStat("mr")
    ROOT.gStyle.SetStatColor(ROOT.kWhite)
    ROOT.gStyle.SetStatFont(42)
    ROOT.gStyle.SetStatFontSize(0.025)
    ROOT.gStyle.SetStatTextColor(1)
    ROOT.gStyle.SetStatFormat("6.4g")
    ROOT.gStyle.SetStatBorderSize(1)
    ROOT.gStyle.SetStatH(0.1)
    ROOT.gStyle.SetStatW(0.15)

    ROOT.gStyle.SetHatchesSpacing(1.3/math.sqrt(cvscale))
    ROOT.gStyle.SetHatchesLineWidth(int(2*cvscale))

    # ROOT.gStyle.SetStaROOT.TStyle(Style_t style = 1001)
    # ROOT.gStyle.SetStatX(Float_t x = 0)
    # ROOT.gStyle.SetStatY(Float_t y = 0)


    #ROOT.gROOT.ForceStyle(True)
    #end modified

    # For the Global title:

    ROOT.gStyle.SetOptTitle(0)

    # ROOT.gStyle.SetTitleH(0) # Set the height of the title box
    # ROOT.gStyle.SetTitleW(0) # Set the width of the title box
    #ROOT.gStyle.SetTitleX(0.35) # Set the position of the title box
    #ROOT.gStyle.SetTitleY(0.986) # Set the position of the title box
    # ROOT.gStyle.SetTitleStyle(Style_t style = 1001)
    #ROOT.gStyle.SetTitleBorderSize(0)

    # For the axis titles:
    ROOT.gStyle.SetTitleColor(1, "XYZ")
    ROOT.gStyle.SetTitleFont(43, "XYZ")
    ROOT.gStyle.SetTitleSize(35*cvscale*fontScale, "XYZ")
    # ROOT.gStyle.SetTitleXSize(Float_t size = 0.02) # Another way to set the size?
    # ROOT.gStyle.SetTitleYSize(Float_t size = 0.02)
    ROOT.gStyle.SetTitleXOffset(1.2)
    #ROOT.gStyle.SetTitleYOffset(1.2)
    ROOT.gStyle.SetTitleOffset(1.2, "YZ") # Another way to set the Offset

    # For the axis labels:

    ROOT.gStyle.SetLabelColor(1, "XYZ")
    ROOT.gStyle.SetLabelFont(43, "XYZ")
    ROOT.gStyle.SetLabelOffset(0.0077, "XYZ")
    ROOT.gStyle.SetLabelSize(32*cvscale*fontScale, "XYZ")
    #ROOT.gStyle.SetLabelSize(0.04, "XYZ")

    # For the axis:

    ROOT.gStyle.SetAxisColor(1, "XYZ")
    ROOT.gStyle.SetAxisColor(1, "XYZ")
    ROOT.gStyle.SetStripDecimals(True)
    ROOT.gStyle.SetTickLength(0.03, "Y")
    ROOT.gStyle.SetTickLength(0.05, "X")
    ROOT.gStyle.SetNdivisions(1005, "X")
    ROOT.gStyle.SetNdivisions(506, "Y")

    ROOT.gStyle.SetPadTickX(1)  # To get tick marks on the opposite side of the frame
    ROOT.gStyle.SetPadTickY(1)

    # Change for log plots:
    ROOT.gStyle.SetOptLogx(0)
    ROOT.gStyle.SetOptLogy(0)
    ROOT.gStyle.SetOptLogz(0)

    #ROOT.gStyle.SetPalette(1) #(1,0)

    # another top group addition

    # Postscript options:
    #ROOT.gStyle.SetPaperSize(20., 20.)
    #ROOT.gStyle.SetPaperSize(ROOT.TStyle.kA4)
    #ROOT.gStyle.SetPaperSize(27., 29.7)
    #ROOT.gStyle.SetPaperSize(27., 29.7)
    ROOT.gStyle.SetPaperSize(8.0*1.6*cvscale,7.0*1.6*cvscale)
    ROOT.TGaxis.SetMaxDigits(3)
    ROOT.gStyle.SetLineScalePS(2)

    # ROOT.gStyle.SetLineStyleString(Int_t i, const char* text)
    # ROOT.gStyle.SetHeaderPS(const char* header)
    # ROOT.gStyle.SetTitlePS(const char* pstitle)
    #ROOT.gStyle.SetColorModelPS(1)

    # ROOT.gStyle.SetBarOffset(Float_t baroff = 0.5)
    # ROOT.gStyle.SetBarWidth(Float_t barwidth = 0.5)
    # ROOT.gStyle.SetPaintTextFormat(const char* format = "g")
    # ROOT.gStyle.SetPalette(Int_t ncolors = 0, Int_t* colors = 0)
    # ROOT.gStyle.SetTimeOffset(Double_t toffset)
    # ROOT.gStyle.SetHistMinimumZero(kTRUE)

    ROOT.gStyle.SetPaintTextFormat("3.0f")

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

def makePlot(outputFolder, histDict,branchNameList,binning,title,output,target=None,logx=0,logy=0):
    cv = ROOT.TCanvas("cv","",1100,700)
    cv.SetLogx(logx)
    cv.SetLogy(logy)
    cv.SetRightMargin(0.36)
    ymax = max(map(lambda h: h.GetMaximum(),histDict.values()))
    axis = ROOT.TH2F("axis",title,50,binning[0],binning[-1],50,0,ymax*1.1)
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

def getROC(signal,background):
    N=signal.GetNbinsX()+2
    sigHistContent=np.zeros(N)
    bgHistContent=np.zeros(N)
    
    for i in range(N):
        sigHistContent[i]=signal.GetBinContent(i)
        bgHistContent[i]=background.GetBinContent(i)

    sigN=sum(sigHistContent)
    bgN=sum(bgHistContent)

    sigEff=[]
    bgRej=[]
    bgEff=[]
    sig_integral=0.0
    bg_integral=0.0

    for ibin in reversed(range(N)):
        sig_integral+=sigHistContent[ibin]
        bg_integral+=bgHistContent[ibin]
        sigEff.append(sig_integral/sigN)
        bgRej.append(1-bg_integral/bgN)
        bgEff.append(bg_integral/bgN)
    return sigEff,bgRej,bgEff
    
def getAUC(sigEff,bgRej):
    integral=0.0
    for i in range(len(sigEff)-1):
        w=math.fabs(sigEff[i+1]-sigEff[i])
        h=0.5*(bgRej[i+1]+bgRej[i])
        x=(sigEff[i+1]+sigEff[i])*0.5
        integral+=w*math.fabs(h-(1-x))
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

    axis=ROOT.TH2F("axis"+str(random.random()),";"+signalName+" efficiency;"+backgroundName+" efficiency",50,0,1.0,50,0.0008,1.0)
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
    #cv.Print(name+".png")
    cv.WaitPrimitive()
