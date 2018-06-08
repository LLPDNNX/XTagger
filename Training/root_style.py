import ROOT
import math

def set_root_style():

    cvscale = 1.
    fontScale = 750./650.

    NCont = 255;

    ROOT.gStyle.SetNumberContours(NCont)
    ROOT.gRandom.SetSeed(123)
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
