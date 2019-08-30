import ROOT

'''
some style suggestions
* canvas size: 800 x 670
* general text font: 43
* axis titel size: 33
* axis label size: 29
* 'CMS' text font: 63, size: 31
* additional 'Simulation' text font: 53, size: 31
* text next to 'CMS' (e.g. lumi) size: 31
* line width (e.g. ROC curves): at least 2
* legend text size: 27 - 29
'''

ROOT.gROOT.SetStyle("Plain")
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptDate(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFile(0)
ROOT.gStyle.SetOptTitle(0)

ROOT.gStyle.SetCanvasBorderMode(0)
ROOT.gStyle.SetCanvasColor(ROOT.kWhite)

ROOT.gStyle.SetPadBorderMode(0)
ROOT.gStyle.SetPadColor(ROOT.kWhite)
ROOT.gStyle.SetGridColor(ROOT.kBlack)
ROOT.gStyle.SetGridStyle(2)
ROOT.gStyle.SetGridWidth(1)

ROOT.gStyle.SetFrameBorderMode(0)
ROOT.gStyle.SetFrameBorderSize(0)
ROOT.gStyle.SetFrameFillColor(0)
ROOT.gStyle.SetFrameFillStyle(0)
ROOT.gStyle.SetFrameLineColor(1)
ROOT.gStyle.SetFrameLineStyle(1)
ROOT.gStyle.SetFrameLineWidth(0)

ROOT.gStyle.SetEndErrorSize(2)
ROOT.gStyle.SetErrorX(0.)
ROOT.gStyle.SetMarkerStyle(20)

ROOT.gStyle.SetHatchesSpacing(0.9)
ROOT.gStyle.SetHatchesLineWidth(2)

ROOT.gStyle.SetTitleColor(1, "XYZ")
ROOT.gStyle.SetTitleFont(43, "XYZ")
ROOT.gStyle.SetTitleSize(33, "XYZ")
ROOT.gStyle.SetTitleXOffset(1.135)
ROOT.gStyle.SetTitleOffset(1.32, "YZ")

ROOT.gStyle.SetLabelColor(1, "XYZ")
ROOT.gStyle.SetLabelFont(43, "XYZ")
ROOT.gStyle.SetLabelSize(29, "XYZ")

ROOT.gStyle.SetAxisColor(1, "XYZ")
ROOT.gStyle.SetAxisColor(1, "XYZ")
ROOT.gStyle.SetStripDecimals(True)
ROOT.gStyle.SetNdivisions(1005, "X")
ROOT.gStyle.SetNdivisions(506, "Y")

ROOT.gStyle.SetPadTickX(1)
ROOT.gStyle.SetPadTickY(1)

ROOT.gStyle.SetPaperSize(8.0*1.35,6.7*1.35)
ROOT.TGaxis.SetMaxDigits(3)
ROOT.gStyle.SetLineScalePS(2)

ROOT.gStyle.SetPalette(1)
ROOT.gStyle.SetPaintTextFormat(".1f")

colors = []
    
def newColorRGB(red,green,blue):
    newColorRGB.colorindex+=1
    color=ROOT.TColor(newColorRGB.colorindex,red,green,blue)
    colors.append(color)
    return color
    
def HLS2RGB(hue,light,sat):
    r, g, b = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    ROOT.TColor.HLS2RGB(
        int(round(hue*255.)),
        int(round(light*255.)),
        int(round(sat*255.)),
        r,g,b
    )
    return r.value/255.,g.value/255.,b.value/255.
    
def newColorHLS(hue,light,sat):
    r,g,b = HLS2RGB(hue,light,sat)
    return newColorRGB(r,g,b)
    
newColorRGB.colorindex=301

def makeColorTable(reverse=False):
    colorList = [
        [0.,newColorHLS(0.6, 0.47,0.6)],
        [0.,newColorHLS(0.56, 0.65, 0.7)],
        [0.,newColorHLS(0.52, 1., 1.)],
    ]
    
    if reverse:
        colorList = reversed(colorList)

    lumiMin = min(map(lambda x:x[1].GetLight(),colorList))
    lumiMax = max(map(lambda x:x[1].GetLight(),colorList))

    for color in colorList:
        if reverse:
            color[0] = ((lumiMax-color[1].GetLight())/(lumiMax-lumiMin))
        else:
            color[0] = ((color[1].GetLight()-lumiMin)/(lumiMax-lumiMin))


    stops = numpy.array(map(lambda x:x[0],colorList))
    red   = numpy.array(map(lambda x:x[1].GetRed(),colorList))
    green = numpy.array(map(lambda x:x[1].GetGreen(),colorList))
    blue  = numpy.array(map(lambda x:x[1].GetBlue(),colorList))

    start=ROOT.TColor.CreateGradientColorTable(len(stops), stops, red, green, blue, 200)
    ROOT.gStyle.SetNumberContours(200)



rootObj = []

def makeCanvas(name="cv",width=800,height=670):
    ROOT.gStyle.SetPaperSize(width*0.0135,height*0.0135)
    cv = ROOT.TCanvas(name,"",width,height)
    rootObj.append(cv)
    return cv

def makeLegend(x1,y1,x2,y2):
    legend = ROOT.TLegend(x1,y1,x2,y2)
    legend.SetBorderSize(0)
    legend.SetTextFont(43)
    legend.SetTextSize(29)
    legend.SetFillStyle(0)
    rootObj.append(legend)
    return legend
    
def makeCMSText(x1,y1,additionalText=None):
    pTextCMS = ROOT.TPaveText(x1,y1,x1,y1,"NDC")
    pTextCMS.AddText("CMS")
    pTextCMS.SetTextFont(63)
    pTextCMS.SetTextSize(31)
    pTextCMS.SetTextAlign(13)
    rootObj.append(pTextCMS)
    pTextCMS.Draw("Same")

    if additionalText:
        pTextAdd = ROOT.TPaveText(x1+0.088,y1,x1+0.088,y1,"NDC")
        pTextAdd.AddText(additionalText)
        pTextAdd.SetTextFont(53)
        pTextAdd.SetTextSize(31)
        pTextAdd.SetTextAlign(13)
        rootObj.append(pTextAdd)
        pTextAdd.Draw("Same")
    
def makeLumiText(x1,y1):
    pText = ROOT.TPaveText(x1,y1,x1,y1,"NDC")
    pText.AddText("36 fb#lower[-0.8]{#scale[0.7]{-1}}")
    pText.SetTextFont(63)
    pText.SetTextSize(31)
    pText.SetTextAlign(33)
    rootObj.append(pTextCMS)
    pText.Draw("Same")

ptSymbol = "p#kern[-0.8]{ }#lower[0.3]{#scale[0.7]{T}}"
metSymbol = ptSymbol+"#kern[-2.3]{ }#lower[-0.8]{#scale[0.7]{miss}}"
metSymbol_lc = ptSymbol+"#kern[-2.3]{ }#lower[-0.8]{#scale[0.7]{miss,#kern[-0.5]{ }#mu-corr.}}}"
minDPhiSymbol = "#Delta#phi#lower[-0.05]{*}#kern[-1.9]{ }#lower[0.3]{#scale[0.7]{min}}"
htSymbol = "H#kern[-0.7]{ }#lower[0.3]{#scale[0.7]{T}}"
mhtSymbol = "H#kern[-0.7]{ }#lower[0.3]{#scale[0.7]{T}}#kern[-2.2]{ }#lower[-0.8]{#scale[0.7]{miss}}"
rSymbol = mhtSymbol+"#lower[0.05]{#scale[1.2]{/}}"+metSymbol
rSymbol_lc = mhtSymbol+"#lower[0.05]{#scale[1.2]{/}}"+metSymbol_lc
mzSymbol = "m#lower[0.3]{#scale[0.7]{#mu#mu}}"
gSymbol = "#tilde{g}"
qbarSymbol = "q#lower[-0.8]{#kern[-0.89]{#minus}}"
mgSymbol = "m#lower[0.2]{#scale[0.8]{#kern[-0.75]{ }"+gSymbol+"}}"
chiSymbol = "#tilde{#chi}#lower[-0.5]{#scale[0.65]{0}}#kern[-1.2]{#lower[0.6]{#scale[0.65]{1}}}"
mchiSymbol = "m#lower[0.2]{#scale[0.8]{"+chiSymbol+"}}"

def ctauSymbol(logctau=-3):
    symbols = [
        ["c#tau#kern[-0.5]{ }=#kern[-0.5]{ }1#kern[-0.6]{ }#mum"],
        ["c#tau#kern[-0.5]{ }=#kern[-0.5]{ }10#kern[-0.6]{ }#mum"],
        ["c#tau#kern[-0.5]{ }=#kern[-0.5]{ }100#kern[-0.6]{ }#mum"],
        ["c#tau#kern[-0.5]{ }=#kern[-0.5]{ }1#kern[-0.6]{ }mm"],
        ["c#tau#kern[-0.5]{ }=#kern[-0.5]{ }10#kern[-0.6]{ }mm"],
        ["c#tau#kern[-0.5]{ }=#kern[-0.5]{ }100#kern[-0.6]{ }mm"],
        ["c#tau#kern[-0.5]{ }=#kern[-0.5]{ }1#kern[-0.6]{ }m"],
        ["c#tau#kern[-0.5]{ }=#kern[-0.5]{ }10#kern[-0.6]{ }m"],
    ]
    return symbols[logctau+3]
    
    
    
