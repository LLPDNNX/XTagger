#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TTreeFormula.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <random>
#include <algorithm>

#include "cmdParser.hpp"
#include "exprtk.hpp"


class UnpackedTree
{
    public:
        const bool addTruth_;
        TFile* outputFile_;
        TTree* tree_;
        
        
        static constexpr int maxEntries = 25;
        static constexpr int bufferSize = 64000; //default is 32kB
        
        unsigned int jetorigin_isPU;
        unsigned int jetorigin_isUndefined;
        
        float jetorigin_displacement;
        float jetorigin_ctau;
        float jetorigin_decay_angle;
        
        unsigned int jetorigin_isB;
        unsigned int jetorigin_isBB;
        unsigned int jetorigin_isGBB;
        unsigned int jetorigin_isLeptonic_B;
        unsigned int jetorigin_isLeptonic_C;
        unsigned int jetorigin_isC;
        unsigned int jetorigin_isCC;
        unsigned int jetorigin_isGCC;
        unsigned int jetorigin_isS;
        unsigned int jetorigin_isUD;
        unsigned int jetorigin_isG;
        unsigned int jetorigin_fromLLP;
        
        float global_pt;
        float global_eta;
        float global_rho;
        
        float isData;
        float xsecweight;
        
        unsigned int ncpf;
        float cpf_trackEtaRel[maxEntries];
        float cpf_trackPtRel[maxEntries];
        float cpf_trackPPar[maxEntries];
        float cpf_trackDeltaR[maxEntries];
        float cpf_trackPtRatio[maxEntries];
        float cpf_trackPParRatio[maxEntries];
        float cpf_trackSip2dVal[maxEntries];
        float cpf_trackSip2dSig[maxEntries];
        float cpf_trackSip3dVal[maxEntries];
        float cpf_trackSip3dSig[maxEntries];
        float cpf_trackJetDistVal[maxEntries];
        float cpf_trackJetDistSig[maxEntries];
        float cpf_ptrel[maxEntries];
        float cpf_drminsv[maxEntries];
        float cpf_vertex_association[maxEntries];
        float cpf_puppi_weight[maxEntries];
        float cpf_track_chi2[maxEntries];
        float cpf_track_quality[maxEntries];
        float cpf_jetmassdroprel[maxEntries];
        float cpf_relIso01[maxEntries];
        
        float csv_trackSumJetEtRatio;
        float csv_trackSumJetDeltaR;
        float csv_vertexCategory;
        float csv_trackSip2dValAboveCharm;
        float csv_trackSip2dSigAboveCharm;
        float csv_trackSip3dValAboveCharm;
        float csv_trackSip3dSigAboveCharm;
        float csv_jetNSelectedTracks;
        float csv_jetNTracksEtaRel;
        
        unsigned int nnpf;
        float npf_ptrel[maxEntries];
        float npf_deltaR[maxEntries];
        float npf_isGamma[maxEntries];
        float npf_hcal_fraction[maxEntries];
        float npf_drminsv[maxEntries];
        float npf_puppi_weight[maxEntries];
        float npf_jetmassdroprel[maxEntries];
        float npf_relIso01[maxEntries];
        
        unsigned int nsv;
        float sv_pt[maxEntries];
        float sv_mass[maxEntries];
        float sv_deltaR[maxEntries];
        float sv_ntracks[maxEntries];
        float sv_chi2[maxEntries];
        float sv_normchi2[maxEntries];
        float sv_dxy[maxEntries];
        float sv_dxysig[maxEntries];
        float sv_d3d[maxEntries];
        float sv_d3dsig[maxEntries];
        float sv_costhetasvpv[maxEntries];
        float sv_enratio[maxEntries];
        
    public:
        UnpackedTree(const std::string& fileName, bool addTruth=true):
            addTruth_(addTruth),
            outputFile_(new TFile(fileName.c_str(),"RECREATE")),
            tree_(new TTree("jets","jets"))
        {

            tree_->SetDirectory(outputFile_);
            tree_->SetAutoSave(200); //save after 200 fills
            
            if (addTruth)
            {
                tree_->Branch("jetorigin_isPU",&jetorigin_isPU,"jetorigin_isPU/I",bufferSize);
                tree_->Branch("jetorigin_isUndefined",&jetorigin_isUndefined,"jetorigin_isUndefined/I",bufferSize);
                
                tree_->Branch("jetorigin_displacement",&jetorigin_displacement,"jetorigin_displacement/F",bufferSize);
                tree_->Branch("jetorigin_ctau",&jetorigin_ctau,"jetorigin_ctau/F",bufferSize);
                tree_->Branch("jetorigin_decay_angle",&jetorigin_decay_angle,"jetorigin_decay_angle/F",bufferSize);
                
                tree_->Branch("jetorigin_isB",&jetorigin_isB,"jetorigin_isB/I",bufferSize);
                tree_->Branch("jetorigin_isBB",&jetorigin_isBB,"jetorigin_isBB/I",bufferSize);
                tree_->Branch("jetorigin_isGBB",&jetorigin_isGBB,"jetorigin_isGBB/I",bufferSize);
                tree_->Branch("jetorigin_isLeptonic_B",&jetorigin_isLeptonic_B,"jetorigin_isLeptonic_B/I",bufferSize);
                tree_->Branch("jetorigin_isLeptonic_C",&jetorigin_isLeptonic_C,"jetorigin_isLeptonic_C/I",bufferSize);
                tree_->Branch("jetorigin_isC",&jetorigin_isC,"jetorigin_isC/I",bufferSize);
                tree_->Branch("jetorigin_isCC",&jetorigin_isCC,"jetorigin_isCC/I",bufferSize);
                tree_->Branch("jetorigin_isGCC",&jetorigin_isGCC,"jetorigin_isGCC/I",bufferSize);
                tree_->Branch("jetorigin_isS",&jetorigin_isS,"jetorigin_isS/I",bufferSize);
                tree_->Branch("jetorigin_isUD",&jetorigin_isUD,"jetorigin_isUD/I",bufferSize);
                tree_->Branch("jetorigin_isG",&jetorigin_isG,"jetorigin_isG/I",bufferSize);
                tree_->Branch("jetorigin_fromLLP",&jetorigin_fromLLP,"jetorigin_fromLLP/I",bufferSize);
            }
            else
            {
                tree_->Branch("xsecweight",&xsecweight,"xsecweight/F",bufferSize);
                tree_->Branch("isData",&isData,"isData/F",bufferSize);
            }
            
            tree_->Branch("global_pt",&global_pt,"global_pt/F",bufferSize);
            tree_->Branch("global_eta",&global_eta,"global_eta/F",bufferSize);
            tree_->Branch("global_rho",&global_rho,"global_rho/F",bufferSize);

            tree_->Branch("ncpf",&ncpf,"ncpf/I",bufferSize);
            tree_->Branch("cpf_trackEtaRel",&cpf_trackEtaRel,"cpf_trackEtaRel[ncpf]/F",bufferSize);
            tree_->Branch("cpf_trackPtRel",&cpf_trackPtRel,"cpf_trackPtRel[ncpf]/F",bufferSize);
            tree_->Branch("cpf_trackPPar",&cpf_trackPPar,"cpf_trackPPar[ncpf]/F",bufferSize);
            tree_->Branch("cpf_trackDeltaR",&cpf_trackDeltaR,"cpf_trackDeltaR[ncpf]/F",bufferSize);
            tree_->Branch("cpf_trackPtRatio",&cpf_trackPtRatio,"cpf_trackPtRatio[ncpf]/F",bufferSize);
            tree_->Branch("cpf_trackPParRatio",&cpf_trackPParRatio,"cpf_trackPParRatio[ncpf]/F",bufferSize);
            tree_->Branch("cpf_trackSip2dVal",&cpf_trackSip2dVal,"cpf_trackSip2dVal[ncpf]/F",bufferSize);
            tree_->Branch("cpf_trackSip2dSig",&cpf_trackSip2dSig,"cpf_trackSip2dSig[ncpf]/F",bufferSize);
            tree_->Branch("cpf_trackSip3dVal",&cpf_trackSip3dVal,"cpf_trackSip3dVal[ncpf]/F",bufferSize);
            tree_->Branch("cpf_trackSip3dSig",&cpf_trackSip3dSig,"cpf_trackSip3dSig[ncpf]/F",bufferSize);
            tree_->Branch("cpf_trackJetDistVal",&cpf_trackJetDistVal,"cpf_trackJetDistVal[ncpf]/F",bufferSize);
            tree_->Branch("cpf_trackJetDistSig",&cpf_trackJetDistSig,"cpf_trackJetDistSig[ncpf]/F",bufferSize);
            tree_->Branch("cpf_ptrel",&cpf_ptrel,"cpf_ptrel[ncpf]/F",bufferSize);
            tree_->Branch("cpf_drminsv",&cpf_drminsv,"cpf_drminsv[ncpf]/F",bufferSize);
            tree_->Branch("cpf_vertex_association",&cpf_vertex_association,"cpf_vertex_association[ncpf]/F",bufferSize);
            tree_->Branch("cpf_puppi_weight",&cpf_puppi_weight,"cpf_puppi_weight[ncpf]/F",bufferSize);
            tree_->Branch("cpf_track_chi2",&cpf_track_chi2,"cpf_track_chi2[ncpf]/F",bufferSize);
            tree_->Branch("cpf_track_quality",&cpf_track_quality,"cpf_track_quality[ncpf]/F",bufferSize);
            tree_->Branch("cpf_jetmassdroprel",&cpf_jetmassdroprel,"cpf_jetmassdroprel[ncpf]/F",bufferSize);
            tree_->Branch("cpf_relIso01",&cpf_relIso01,"cpf_relIso01[ncpf]/F",bufferSize);
            
            tree_->Branch("csv_trackSumJetEtRatio",&csv_trackSumJetEtRatio,"csv_trackSumJetEtRatio/F",bufferSize);
            tree_->Branch("csv_trackSumJetDeltaR",&csv_trackSumJetDeltaR,"csv_trackSumJetDeltaR/F",bufferSize);
            tree_->Branch("csv_vertexCategory",&csv_vertexCategory,"csv_vertexCategory/F",bufferSize);
            tree_->Branch("csv_trackSip2dValAboveCharm",&csv_trackSip2dValAboveCharm,"csv_trackSip2dValAboveCharm/F",bufferSize);
            tree_->Branch("csv_trackSip2dSigAboveCharm",&csv_trackSip2dSigAboveCharm,"csv_trackSip2dSigAboveCharm/F",bufferSize);
            tree_->Branch("csv_trackSip3dValAboveCharm",&csv_trackSip3dValAboveCharm,"csv_trackSip3dValAboveCharm/F",bufferSize);
            tree_->Branch("csv_trackSip3dSigAboveCharm",&csv_trackSip3dSigAboveCharm,"csv_trackSip3dSigAboveCharm/F",bufferSize);
            tree_->Branch("csv_jetNSelectedTracks",&csv_jetNSelectedTracks,"csv_jetNSelectedTracks/F",bufferSize);
            tree_->Branch("csv_jetNTracksEtaRel",&csv_jetNTracksEtaRel,"csv_jetNTracksEtaRel/F",bufferSize);

            tree_->Branch("nnpf",&nnpf,"nnpf/I",bufferSize);
            tree_->Branch("npf_ptrel",&npf_ptrel,"npf_ptrel[nnpf]/F",bufferSize);
            tree_->Branch("npf_deltaR",&npf_deltaR,"npf_deltaR[nnpf]/F",bufferSize);
            tree_->Branch("npf_isGamma",&npf_isGamma,"npf_isGamma[nnpf]/F",bufferSize);
            tree_->Branch("npf_hcal_fraction",&npf_hcal_fraction,"npf_hcal_fraction[nnpf]/F",bufferSize);
            tree_->Branch("npf_drminsv",&npf_drminsv,"npf_drminsv[nnpf]/F",bufferSize);
            tree_->Branch("npf_puppi_weight",&npf_puppi_weight,"npf_puppi_weight[nnpf]/F",bufferSize);
            tree_->Branch("npf_jetmassdroprel",&npf_jetmassdroprel,"npf_jetmassdroprel[nnpf]/F",bufferSize);
            tree_->Branch("npf_relIso01",&npf_relIso01,"npf_relIso01[nnpf]/F",bufferSize);

            tree_->Branch("nsv",&nsv,"nsv/I",bufferSize);
            tree_->Branch("sv_pt",&sv_pt,"sv_pt[nsv]/F",bufferSize);
            tree_->Branch("sv_mass",&sv_deltaR,"sv_mass[nsv]/F",bufferSize);
            tree_->Branch("sv_deltaR",&sv_deltaR,"sv_deltaR[nsv]/F",bufferSize);
            tree_->Branch("sv_ntracks",&sv_ntracks,"sv_ntracks[nsv]/F",bufferSize);
            tree_->Branch("sv_chi2",&sv_chi2,"sv_chi2[nsv]/F",bufferSize);
            tree_->Branch("sv_normchi2",&sv_normchi2,"sv_normchi2[nsv]/F",bufferSize);
            tree_->Branch("sv_dxy",&sv_dxy,"sv_dxy[nsv]/F",bufferSize);
            tree_->Branch("sv_dxysig",&sv_dxysig,"sv_dxysig[nsv]/F",bufferSize);
            tree_->Branch("sv_d3d",&sv_d3d,"sv_d3d[nsv]/F",bufferSize);
            tree_->Branch("sv_d3dsig",&sv_d3dsig,"sv_d3dsig[nsv]/F",bufferSize);
            tree_->Branch("sv_costhetasvpv",&sv_costhetasvpv,"sv_costhetasvpv[nsv]/F",bufferSize);
            tree_->Branch("sv_enratio",&sv_enratio,"sv_enratio[nsv]/F",bufferSize);
        
            tree_->SetBasketSize("*",bufferSize); //default is 16kB
        }
        
        //root does not behave properly
        UnpackedTree(UnpackedTree&&) = delete;
        UnpackedTree(const UnpackedTree&) = delete;
        
        ~UnpackedTree()
        {
            if (outputFile_)
            {
                delete outputFile_;
            }
            //Note: TTree is managed by TFile and gets deleted by ROOT when file is closed
        }
        
        
        void fill()
        {
            //outputFile_->cd();
            //tree_->SetDirectory(outputFile_);
            tree_->Fill();
        }
        
        void close()
        {
            outputFile_->cd();
            //tree_->SetDirectory(outputFile_);
            tree_->Write();
            outputFile_->Close();
        }
};

class NanoXTree
{
    public:
        //std::shared_ptr<TFile> file_;
        TTree* tree_;
        const bool addTruth_;
        
        unsigned int ientry_;
        
        static constexpr int maxEntries = 250; //25*10 -> allows for a maximum of 10 jets per event
        
        unsigned int nJet;
        float Jet_eta[maxEntries];
        float Jet_pt[maxEntries];
        unsigned int Jet_jetId[maxEntries];
        unsigned int Jet_nConstituents[maxEntries];
        unsigned int Jet_cleanmask[maxEntries];
        unsigned int Jet_forDA[maxEntries];
        
        unsigned int njetorigin;
        float jetorigin_isPU[maxEntries];
        float jetorigin_isUndefined[maxEntries];
        
        float jetorigin_displacement[maxEntries];
        float jetorigin_decay_angle[maxEntries];
        
        float jetorigin_isB[maxEntries];
        float jetorigin_isBB[maxEntries];
        float jetorigin_isGBB[maxEntries];
        float jetorigin_isLeptonic_B[maxEntries];
        float jetorigin_isLeptonic_C[maxEntries];
        float jetorigin_isC[maxEntries];
        float jetorigin_isCC[maxEntries];
        float jetorigin_isGCC[maxEntries];
        float jetorigin_isS[maxEntries];
        float jetorigin_isUD[maxEntries];
        float jetorigin_isG[maxEntries];
        float jetorigin_fromLLP[maxEntries];
        
        //float jetorigin_llpmass_reco[maxEntries];
        
        unsigned int nglobal;
        float global_pt[maxEntries];
        float global_eta[maxEntries];
        float global_rho;
        float xsecweight;
        float isData;
        
        unsigned int ncpflength;
        float cpflength_length[maxEntries];
        
        unsigned int ncpf[maxEntries];
        float cpf_trackEtaRel[maxEntries];
        float cpf_trackPtRel[maxEntries];
        float cpf_trackPPar[maxEntries];
        float cpf_trackDeltaR[maxEntries];
        float cpf_trackPtRatio[maxEntries];
        float cpf_trackPParRatio[maxEntries];
        float cpf_trackSip2dVal[maxEntries];
        float cpf_trackSip2dSig[maxEntries];
        float cpf_trackSip3dVal[maxEntries];
        float cpf_trackSip3dSig[maxEntries];
        float cpf_trackJetDistVal[maxEntries];
        float cpf_trackJetDistSig[maxEntries];
        float cpf_ptrel[maxEntries];
        float cpf_drminsv[maxEntries];
        float cpf_vertex_association[maxEntries];
        float cpf_puppi_weight[maxEntries];
        float cpf_track_chi2[maxEntries];
        float cpf_track_quality[maxEntries];
        float cpf_jetmassdroprel[maxEntries];
        float cpf_relIso01[maxEntries];
        
        unsigned int ncsv[maxEntries];
        float csv_trackSumJetEtRatio[maxEntries];
        float csv_trackSumJetDeltaR[maxEntries];
        float csv_vertexCategory[maxEntries];
        float csv_trackSip2dValAboveCharm[maxEntries];
        float csv_trackSip2dSigAboveCharm[maxEntries];
        float csv_trackSip3dValAboveCharm[maxEntries];
        float csv_trackSip3dSigAboveCharm[maxEntries];
        float csv_jetNSelectedTracks[maxEntries];
        float csv_jetNTracksEtaRel[maxEntries];
        
        unsigned int nnpflength;
        float npflength_length[maxEntries];
        
        unsigned int nnpf[maxEntries];
        float npf_ptrel[maxEntries];
        float npf_deltaR[maxEntries];
        float npf_isGamma[maxEntries];
        float npf_hcal_fraction[maxEntries];
        float npf_drminsv[maxEntries];
        float npf_puppi_weight[maxEntries];
        float npf_jetmassdroprel[maxEntries];
        float npf_relIso01[maxEntries];
        
        unsigned int nsvlength;
        float svlength_length[maxEntries];
        
        unsigned int nsv[maxEntries];
        float sv_pt[maxEntries];
        float sv_mass[maxEntries];
        float sv_deltaR[maxEntries];
        float sv_ntracks[maxEntries];
        float sv_chi2[maxEntries];
        float sv_normchi2[maxEntries];
        float sv_dxy[maxEntries];
        float sv_dxysig[maxEntries];
        float sv_d3d[maxEntries];
        float sv_d3dsig[maxEntries];
        float sv_costhetasvpv[maxEntries];
        float sv_enratio[maxEntries];
        
        
        std::mt19937 randomGenerator_;
        std::uniform_real_distribution<> uniform_dist_;
        
        typedef exprtk::symbol_table<float> SymbolTable;
        typedef exprtk::expression<float> Expression;
        typedef exprtk::parser<float> Parser;
        
        //for the symbol table
        float isB;
        float isBB;
        float isGBB;
        float isLeptonic_B;
        float isLeptonic_C;
        float isC;
        float isCC;
        float isGCC;
        float isS;
        float isUD;
        float isG;
        float fromLLP;
        
        float rand;
        float logpt;
        float ctau;
        
        SymbolTable symbolTable_;
        std::vector<Expression> selections_;
        std::vector<Expression> setters_;
        
    public:
        NanoXTree(
            TTree* tree, 
            const std::vector<std::string>& selectors={}, 
            const std::vector<std::string>& setters={},
            bool addTruth=true
        ):
            tree_(tree),
            addTruth_(addTruth),
            randomGenerator_(12345),
            uniform_dist_(0,1.)
        {
            tree_->SetBranchAddress("nJet",&nJet);
            tree_->SetBranchAddress("Jet_eta",&Jet_eta);
            tree_->SetBranchAddress("Jet_pt",&Jet_pt);
            tree_->SetBranchAddress("Jet_jetId",&Jet_jetId);
            tree_->SetBranchAddress("Jet_cleanmask",&Jet_cleanmask);
            tree_->SetBranchAddress("Jet_nConstituents",&Jet_nConstituents);
        
            if (addTruth)
            {
                tree_->SetBranchAddress("njetorigin",&njetorigin);
                
                tree_->SetBranchAddress("jetorigin_isPU",&jetorigin_isPU);
                tree_->SetBranchAddress("jetorigin_isUndefined",&jetorigin_isUndefined);
                
                tree_->SetBranchAddress("jetorigin_displacement",&jetorigin_displacement);
                tree_->SetBranchAddress("jetorigin_decay_angle",&jetorigin_decay_angle);
                
                tree_->SetBranchAddress("jetorigin_isB",&jetorigin_isB);
                tree_->SetBranchAddress("jetorigin_isBB",&jetorigin_isBB);
                tree_->SetBranchAddress("jetorigin_isGBB",&jetorigin_isGBB);
                tree_->SetBranchAddress("jetorigin_isLeptonic_B",&jetorigin_isLeptonic_B);
                tree_->SetBranchAddress("jetorigin_isLeptonic_C",&jetorigin_isLeptonic_C);
                tree_->SetBranchAddress("jetorigin_isC",&jetorigin_isC);
                tree_->SetBranchAddress("jetorigin_isCC",&jetorigin_isCC);
                tree_->SetBranchAddress("jetorigin_isGCC",&jetorigin_isGCC);
                tree_->SetBranchAddress("jetorigin_isS",&jetorigin_isS);
                tree_->SetBranchAddress("jetorigin_isUD",&jetorigin_isUD);
                tree_->SetBranchAddress("jetorigin_isG",&jetorigin_isG);
                tree_->SetBranchAddress("jetorigin_fromLLP",&jetorigin_fromLLP);
            }
            else
            {
                tree_->SetBranchAddress("Jet_forDA",&Jet_forDA);
                tree_->SetBranchAddress("xsecweight",&xsecweight);
                tree_->SetBranchAddress("isData",&isData);
            }
            //tree_->SetBranchAddress("jetorigin_llpmass_reco",&jetorigin_llpmass_reco);
            
            tree_->SetBranchAddress("nglobal",&nglobal);
            tree_->SetBranchAddress("global_pt",&global_pt);
            tree_->SetBranchAddress("global_eta",&global_eta);
            tree_->SetBranchAddress("fixedGridRhoFastjetAll",&global_rho);
            
            tree_->SetBranchAddress("ncpflength",&ncpflength);
            tree_->SetBranchAddress("cpflength_length",&cpflength_length);
            
            tree_->SetBranchAddress("ncpf",&ncpf);
            tree_->SetBranchAddress("cpf_trackEtaRel",&cpf_trackEtaRel);
            tree_->SetBranchAddress("cpf_trackPtRel",&cpf_trackPtRel);
            tree_->SetBranchAddress("cpf_trackPPar",&cpf_trackPPar);
            tree_->SetBranchAddress("cpf_trackDeltaR",&cpf_trackDeltaR);
            tree_->SetBranchAddress("cpf_trackPtRatio",&cpf_trackPtRatio);
            tree_->SetBranchAddress("cpf_trackPParRatio",&cpf_trackPParRatio);
            tree_->SetBranchAddress("cpf_trackSip2dVal",&cpf_trackSip2dVal);
            tree_->SetBranchAddress("cpf_trackSip2dSig",&cpf_trackSip2dSig);
            tree_->SetBranchAddress("cpf_trackSip3dVal",&cpf_trackSip3dVal);
            tree_->SetBranchAddress("cpf_trackSip3dSig",&cpf_trackSip3dSig);
            tree_->SetBranchAddress("cpf_trackJetDistVal",&cpf_trackJetDistVal);
            tree_->SetBranchAddress("cpf_trackJetDistSig",&cpf_trackJetDistSig);
            tree_->SetBranchAddress("cpf_ptrel",&cpf_ptrel);
            tree_->SetBranchAddress("cpf_drminsv",&cpf_drminsv);
            tree_->SetBranchAddress("cpf_vertex_association",&cpf_vertex_association);
            tree_->SetBranchAddress("cpf_puppi_weight",&cpf_puppi_weight);
            tree_->SetBranchAddress("cpf_track_chi2",&cpf_track_chi2);
            tree_->SetBranchAddress("cpf_track_quality",&cpf_track_quality);
            tree_->SetBranchAddress("cpf_jetmassdroprel",&cpf_jetmassdroprel);
            tree_->SetBranchAddress("cpf_relIso01",&cpf_relIso01);
            
            tree_->SetBranchAddress("ncsv",&ncsv);
            tree_->SetBranchAddress("csv_trackSumJetEtRatio",&csv_trackSumJetEtRatio);
            tree_->SetBranchAddress("csv_trackSumJetDeltaR",&csv_trackSumJetDeltaR);
            tree_->SetBranchAddress("csv_vertexCategory",&csv_vertexCategory);
            tree_->SetBranchAddress("csv_trackSip2dValAboveCharm",&csv_trackSip2dValAboveCharm);
            tree_->SetBranchAddress("csv_trackSip2dSigAboveCharm",&csv_trackSip2dSigAboveCharm);
            tree_->SetBranchAddress("csv_trackSip3dValAboveCharm",&csv_trackSip3dValAboveCharm);
            tree_->SetBranchAddress("csv_trackSip3dSigAboveCharm",&csv_trackSip3dSigAboveCharm);
            tree_->SetBranchAddress("csv_jetNSelectedTracks",&csv_jetNSelectedTracks);
            tree_->SetBranchAddress("csv_jetNTracksEtaRel",&csv_jetNTracksEtaRel);
           
            tree_->SetBranchAddress("nnpflength",&nnpflength);
            tree_->SetBranchAddress("npflength_length",&npflength_length);
            
            tree_->SetBranchAddress("nnpf",&nnpf);
            tree_->SetBranchAddress("npf_ptrel",&npf_ptrel);
            tree_->SetBranchAddress("npf_deltaR",&npf_deltaR);
            tree_->SetBranchAddress("npf_isGamma",&npf_isGamma);
            tree_->SetBranchAddress("npf_hcal_fraction",&npf_hcal_fraction);
            tree_->SetBranchAddress("npf_drminsv",&npf_drminsv);
            tree_->SetBranchAddress("npf_puppi_weight",&npf_puppi_weight);
            tree_->SetBranchAddress("npf_jetmassdroprel",&npf_jetmassdroprel);
            tree_->SetBranchAddress("npf_relIso01",&npf_relIso01);
            
            tree_->SetBranchAddress("nsvlength",&nsvlength);
            tree_->SetBranchAddress("svlength_length",&svlength_length);
            
            tree_->SetBranchAddress("nsv",&nsv);
            tree_->SetBranchAddress("sv_pt",&sv_pt);
            tree_->SetBranchAddress("sv_mass",&sv_mass);
            tree_->SetBranchAddress("sv_deltaR",&sv_deltaR);
            tree_->SetBranchAddress("sv_ntracks",&sv_ntracks);
            tree_->SetBranchAddress("sv_chi2",&sv_chi2);
            tree_->SetBranchAddress("sv_normchi2",&sv_normchi2);
            tree_->SetBranchAddress("sv_dxy",&sv_dxy);
            tree_->SetBranchAddress("sv_dxysig",&sv_dxysig);
            tree_->SetBranchAddress("sv_d3d",&sv_d3d);
            tree_->SetBranchAddress("sv_d3dsig",&sv_d3dsig);
            tree_->SetBranchAddress("sv_costhetasvpv",&sv_costhetasvpv);
            tree_->SetBranchAddress("sv_enratio",&sv_enratio);
            
            getEvent(0,true);

            symbolTable_.add_variable("isB",isB);
            symbolTable_.add_variable("isBB",isBB);
            symbolTable_.add_variable("isGBB",isGBB);
            symbolTable_.add_variable("isLeptonic_B",isLeptonic_B);
            symbolTable_.add_variable("isLeptonic_C",isLeptonic_C);
            
            symbolTable_.add_variable("isC",isC);
            symbolTable_.add_variable("isCC",isCC);
            symbolTable_.add_variable("isGCC",isGCC);
            
            symbolTable_.add_variable("isS",isS);
            symbolTable_.add_variable("isUD",isUD);
            symbolTable_.add_variable("isG",isG);
            
            symbolTable_.add_variable("fromLLP",fromLLP);
            
            symbolTable_.add_variable("rand",rand);
            symbolTable_.add_variable("ctau",ctau);
            
            symbolTable_.add_variable("logpt",logpt);
            
            for (auto selectstring: selectors)
            {
                std::cout<<"register selection: "<<selectstring<<std::endl;
                Expression exp;
                exp.register_symbol_table(symbolTable_);
                Parser parser;
                parser.compile(selectstring,exp);
                selections_.emplace_back(std::move(exp));
            }
            
            for (auto setstring: setters)
            {
                std::cout<<"register setter: "<<setstring<<std::endl;
                Expression exp;
                exp.register_symbol_table(symbolTable_);
                Parser parser;
                parser.compile(setstring,exp);
                setters_.emplace_back(std::move(exp));
            }
        }
        
        //this class does not play well with memory -> prevent funny usage
        NanoXTree(NanoXTree&&) = delete;
        NanoXTree(const NanoXTree&) = delete;
        
        ~NanoXTree()
        {
            //file_->Close();
        }

        inline unsigned int entries() const
        {
            return tree_->GetEntries();
        }
        
        inline unsigned int entry() const
        {
            return ientry_;
        }
        
        bool getEvent(unsigned int entry, bool force=false)
        {
            if (force or entry!=ientry_)
            {
                tree_->GetEntry(entry);
                ientry_ = entry;
                return true;
            }
            if (entry>=entries())
            {
                return false;
            }
            return true;
        }
        
        bool nextEvent()
        {
            return getEvent(ientry_+1);
        }
       
        inline int njets()
        {
            return nglobal;
        }
        
        bool isSelected(unsigned int jet)
        {
            
            //nJet should be lower than e.g. njetorigin since pT selection on Jet's are applied
            if (jet>=nJet)
            {
                return false;
            }
            
            //just a sanity check
            if (std::fabs(Jet_eta[jet]/global_eta[jet]-1)>0.01)
            {
                std::cout<<"Encountered mismatch between standard nanoaod jets and xtag info"<<std::endl;
                return false;
            }
            
            if (this->njets()<jet)
            {
                std::cout<<"Not enough jets to unpack"<<std::endl;
                return false;
            }
            
            
            //do not apply jet ID; require at least 2 constituents & no overlap with leptons
            //garbage jets are anyway not considered since training is done on matched jets only
            if (Jet_nConstituents[jet]<2 or Jet_cleanmask[jet]==0)
            {
                return false;
            }
            
            if (Jet_pt[jet]<20. or std::fabs(Jet_eta[jet])>2.4)
            {
                return false;
            }
            
            if (addTruth_)
            {
                if (jetorigin_isUndefined[jet]>0.5)
                {
                    return false;
                }
                
                if (jetorigin_isPU[jet]>0.5)
                {
                    return false;
                }
                
                //setup variables for exp evaluation
                isB = jetorigin_isB[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                isBB = jetorigin_isBB[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                isGBB = jetorigin_isGBB[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                isLeptonic_B = jetorigin_isLeptonic_B[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                isLeptonic_C = jetorigin_isLeptonic_C[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                
                isC = jetorigin_isC[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                isCC = jetorigin_isCC[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                isGCC = jetorigin_isGCC[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                
                isS = jetorigin_isS[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                isUD = jetorigin_isUD[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                isG = jetorigin_isG[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                
                fromLLP = jetorigin_fromLLP[jet]>0.5;
            }
            else
            {
                if (Jet_forDA[jet]<0.5)
                {
                    return false;
                }
                isB = 0;
                isBB = 0;
                isGBB = 0;
                isLeptonic_B = 0;
                isLeptonic_C = 0;
                isC = 0;
                isCC = 0;
                isGCC = 0;
                isS = 0;
                isUD = 0;
                isG = 0;
                fromLLP = 0;
            }
            
            
            rand = uniform_dist_(randomGenerator_);
            ctau = 1e9;
            logpt = global_pt[jet];
            
            for (auto setter: setters_)
            {
                //std::cout<<ctau;
                setter.value();
                //std::cout<<" -> "<<ctau<<std::endl;
            }
            
            for (auto exp: selections_)
            {
                if (exp.value()<0.5)
                {
                    return false;
                }
            }

            return true;
        }
        
        int getJetClass(unsigned int jet)
        {
            if (not addTruth_) return 0; //default class
            //if (jetorigin_isPU[jet]>0.5) return 11;
            if (jetorigin_fromLLP[jet]<0.5)
            {
                if  (jetorigin_isB[jet]>0.5) return 0;
                if  (jetorigin_isBB[jet]>0.5) return 1;
                if  (jetorigin_isGBB[jet]>0.5) return 2;
                if  (jetorigin_isLeptonic_B[jet]>0.5) return 3;
                if  (jetorigin_isLeptonic_C[jet]>0.5) return 4;
                if  (jetorigin_isC[jet]>0.5) return 5;
                if  (jetorigin_isCC[jet]>0.5) return 6;
                if  (jetorigin_isGCC[jet]>0.5) return 7;
                if  (jetorigin_isS[jet]>0.5) return 8;
                if  (jetorigin_isUD[jet]>0.5) return 9;
                if  (jetorigin_isG[jet]>0.5) return 10;
            }
            else
            {
                return 11;
            }
            return -1;
        }
        
        bool unpackJet(
            unsigned int jet,
            UnpackedTree& unpackedTree
        )
        {
            //tree_->GetEntry(entry);

            if (this->njets()!=ncpflength or this->njets()!=nnpflength or this->njets()!=nsvlength)
            {
                std::cout<<"Encountered weird event with unclear numbers of jets"<<std::endl;
                std::cout<<"\tnjets = "<<this->njets()<<std::endl;
                std::cout<<"\tnglobal = "<<nglobal<<std::endl;
                std::cout<<"\tncpflength = "<<ncpflength<<std::endl;
                std::cout<<"\tnnpflength = "<<nnpflength<<std::endl;
                std::cout<<"\tnsvlength = "<<nsvlength<<std::endl;
                if (addTruth_)
                {
                    std::cout<<"\tnjetorigin = "<<njetorigin<<std::endl;
                }

                return false;
            }
            
            if (addTruth_)
            {
                unpackedTree.jetorigin_isPU = jetorigin_isPU[jet];
                unpackedTree.jetorigin_isUndefined = jetorigin_isUndefined[jet];
                
                unpackedTree.jetorigin_displacement = jetorigin_displacement[jet];
                unpackedTree.jetorigin_ctau = ctau;
                unpackedTree.jetorigin_decay_angle = jetorigin_decay_angle[jet];
                
                //make DJ and LLP categories exclusive
                unpackedTree.jetorigin_isB = jetorigin_isB[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                unpackedTree.jetorigin_isBB = jetorigin_isBB[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                unpackedTree.jetorigin_isGBB = jetorigin_isGBB[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                unpackedTree.jetorigin_isLeptonic_B = jetorigin_isLeptonic_B[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                unpackedTree.jetorigin_isLeptonic_C = jetorigin_isLeptonic_C[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                unpackedTree.jetorigin_isC = jetorigin_isC[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                unpackedTree.jetorigin_isCC = jetorigin_isCC[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                unpackedTree.jetorigin_isGCC = jetorigin_isGCC[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                unpackedTree.jetorigin_isS = jetorigin_isS[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                unpackedTree.jetorigin_isUD = jetorigin_isUD[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                unpackedTree.jetorigin_isG = jetorigin_isG[jet]>0.5 and jetorigin_fromLLP[jet]<0.5;
                unpackedTree.jetorigin_fromLLP = jetorigin_fromLLP[jet]>0.5;
            }
            else
            {
                unpackedTree.isData = isData;
                unpackedTree.xsecweight = xsecweight;
            }
            
            
            unpackedTree.global_pt = global_pt[jet];
            unpackedTree.global_eta = global_eta[jet];
            unpackedTree.global_rho = global_rho;
            
            int cpf_offset = 0;
            for (size_t i = 0; i < jet; ++i)
            {
                cpf_offset += cpflength_length[i];
            }
            
            unpackedTree.ncpf = cpflength_length[jet];
            int ncpf = std::min<int>(25,cpflength_length[jet]);
            for (size_t i = 0; i < ncpf; ++i)
            {
                unpackedTree.cpf_trackEtaRel[i] = cpf_trackEtaRel[cpf_offset+i];
                unpackedTree.cpf_trackPtRel[i] = cpf_trackPtRel[cpf_offset+i];
                unpackedTree.cpf_trackPPar[i] = cpf_trackPPar[cpf_offset+i];
                unpackedTree.cpf_trackDeltaR[i] = cpf_trackDeltaR[cpf_offset+i];
                unpackedTree.cpf_trackPtRatio[i] = cpf_trackPtRatio[cpf_offset+i];
                unpackedTree.cpf_trackPParRatio[i] = cpf_trackPParRatio[cpf_offset+i];
                unpackedTree.cpf_trackSip2dVal[i] = cpf_trackSip2dVal[cpf_offset+i];
                unpackedTree.cpf_trackSip2dSig[i] = cpf_trackSip2dSig[cpf_offset+i];
                unpackedTree.cpf_trackSip3dVal[i] = cpf_trackSip3dVal[cpf_offset+i];
                unpackedTree.cpf_trackSip3dSig[i] = cpf_trackSip3dSig[cpf_offset+i];
                unpackedTree.cpf_trackJetDistVal[i] = cpf_trackJetDistVal[cpf_offset+i];
                unpackedTree.cpf_trackJetDistSig[i] = cpf_trackJetDistSig[cpf_offset+i];
                unpackedTree.cpf_ptrel[i] = cpf_ptrel[cpf_offset+i];
                unpackedTree.cpf_drminsv[i] = cpf_drminsv[cpf_offset+i];
                unpackedTree.cpf_vertex_association[i] = cpf_vertex_association[cpf_offset+i];
                unpackedTree.cpf_puppi_weight[i] = cpf_puppi_weight[cpf_offset+i];
                unpackedTree.cpf_track_chi2[i] = cpf_track_chi2[cpf_offset+i];
                unpackedTree.cpf_track_quality[i] = cpf_track_quality[cpf_offset+i];
                unpackedTree.cpf_jetmassdroprel[i] = cpf_jetmassdroprel[cpf_offset+i];
                unpackedTree.cpf_relIso01[i] = cpf_relIso01[cpf_offset+i];
            }
            for (size_t i = ncpf; i < 25; ++i)
            {
                unpackedTree.cpf_trackEtaRel[i] = 0;
                unpackedTree.cpf_trackPtRel[i] = 0;
                unpackedTree.cpf_trackPPar[i] = 0;
                unpackedTree.cpf_trackDeltaR[i] = 0;
                unpackedTree.cpf_trackPtRatio[i] = 0;
                unpackedTree.cpf_trackPParRatio[i] = 0;
                unpackedTree.cpf_trackSip2dVal[i] = 0;
                unpackedTree.cpf_trackSip2dSig[i] = 0;
                unpackedTree.cpf_trackSip3dVal[i] = 0;
                unpackedTree.cpf_trackSip3dSig[i] = 0;
                unpackedTree.cpf_trackJetDistVal[i] = 0;
                unpackedTree.cpf_trackJetDistSig[i] = 0;
                unpackedTree.cpf_ptrel[i] = 0;
                unpackedTree.cpf_drminsv[i] = 0;
                unpackedTree.cpf_vertex_association[i] = 0;
                unpackedTree.cpf_puppi_weight[i] = 0;
                unpackedTree.cpf_track_chi2[i] = 0;
                unpackedTree.cpf_track_quality[i] = 0;
                unpackedTree.cpf_jetmassdroprel[i] = 0;
                unpackedTree.cpf_relIso01[i] = 0;
            }
            
            unpackedTree.csv_trackSumJetEtRatio = csv_trackSumJetEtRatio[jet];
            
            unpackedTree.csv_trackSumJetDeltaR = csv_trackSumJetDeltaR[jet];
            unpackedTree.csv_vertexCategory = csv_vertexCategory[jet];
            unpackedTree.csv_trackSip2dValAboveCharm = csv_trackSip2dValAboveCharm[jet];
            unpackedTree.csv_trackSip2dSigAboveCharm = csv_trackSip2dSigAboveCharm[jet];
            unpackedTree.csv_trackSip3dValAboveCharm = csv_trackSip3dValAboveCharm[jet];
            unpackedTree.csv_trackSip3dSigAboveCharm = csv_trackSip3dSigAboveCharm[jet];
            unpackedTree.csv_jetNSelectedTracks = csv_jetNSelectedTracks[jet];
            unpackedTree.csv_jetNTracksEtaRel = csv_jetNTracksEtaRel[jet];
        
        
            int npf_offset = 0;
            for (size_t i = 0; i < jet; ++i)
            {
                npf_offset += npflength_length[i];
            }
            
            unpackedTree.nnpf = npflength_length[jet];
            int nnpf = std::min<int>(25,npflength_length[jet]);
            for (size_t i = 0; i < nnpf; ++i)
            {
                unpackedTree.npf_ptrel[i] = npf_ptrel[npf_offset+i];
                unpackedTree.npf_deltaR[i] = npf_deltaR[npf_offset+i];
                unpackedTree.npf_isGamma[i] = npf_isGamma[npf_offset+i];
                unpackedTree.npf_hcal_fraction[i] = npf_hcal_fraction[npf_offset+i];
                unpackedTree.npf_drminsv[i] = npf_drminsv[npf_offset+i];
                unpackedTree.npf_puppi_weight[i] = npf_puppi_weight[npf_offset+i];
                unpackedTree.npf_jetmassdroprel[i] = npf_jetmassdroprel[npf_offset+i];
                unpackedTree.npf_relIso01[i] = npf_relIso01[npf_offset+i];
            }
            for (size_t i = nnpf; i < 25; ++i)
            {
                unpackedTree.npf_ptrel[i] = 0;
                unpackedTree.npf_deltaR[i] = 0;
                unpackedTree.npf_isGamma[i] = 0;
                unpackedTree.npf_hcal_fraction[i] = 0;
                unpackedTree.npf_drminsv[i] = 0;
                unpackedTree.npf_puppi_weight[i] = 0;
                unpackedTree.npf_jetmassdroprel[i] = 0;
                unpackedTree.npf_relIso01[i] = 0;
            }
            

            int sv_offset = 0;
            for (size_t i = 0; i < jet; ++i)
            {
                sv_offset += svlength_length[i];
            }
            
            unpackedTree.nsv = svlength_length[jet];
            int nsv = std::min<int>(4,svlength_length[jet]);
            for (size_t i = 0; i < nsv; ++i)
            {
                unpackedTree.sv_pt[i] = sv_pt[sv_offset+i];
                unpackedTree.sv_mass[i] = sv_mass[sv_offset+i];
                unpackedTree.sv_deltaR[i] = sv_deltaR[sv_offset+i];
                unpackedTree.sv_ntracks[i] = sv_ntracks[sv_offset+i];
                unpackedTree.sv_chi2[i] = sv_chi2[sv_offset+i];
                unpackedTree.sv_normchi2[i] = sv_normchi2[sv_offset+i];
                unpackedTree.sv_dxy[i] = sv_dxy[sv_offset+i];
                unpackedTree.sv_dxysig[i] = sv_dxysig[sv_offset+i];
                unpackedTree.sv_d3d[i] = sv_d3d[sv_offset+i];
                unpackedTree.sv_d3dsig[i] = sv_d3dsig[sv_offset+i];
                unpackedTree.sv_costhetasvpv[i] = sv_costhetasvpv[sv_offset+i];
                unpackedTree.sv_enratio[i] = sv_enratio[sv_offset+i];
            }
            
            for (size_t i = nsv; i < 4; ++i)
            {
                unpackedTree.sv_pt[i] = 0;
                unpackedTree.sv_mass[i] = 0;
                unpackedTree.sv_deltaR[i] = 0;
                unpackedTree.sv_ntracks[i] = 0;
                unpackedTree.sv_chi2[i] = 0;
                unpackedTree.sv_normchi2[i] = 0;
                unpackedTree.sv_dxy[i] = 0;
                unpackedTree.sv_dxysig[i] = 0;
                unpackedTree.sv_d3d[i] = 0;
                unpackedTree.sv_d3dsig[i] = 0;
                unpackedTree.sv_costhetasvpv[i] = 0;
                unpackedTree.sv_enratio[i] = 0;
            }
            
            unpackedTree.fill();
            return true;
        }
       
};

void printSyntax()
{
    std::cout<<"Syntax: "<<std::endl;
    std::cout<<"          unpackNanoX outputfile N testPercentage Nsplit isplit infile [infile [infile ...]]"<<std::endl<<std::endl;
}

inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

inline bool begins_with(std::string const & value, std::string const & start)
{
    if (start.size() > value.size()) return false;
    return std::equal(start.begin(), start.end(), value.begin());
}

int main(int argc, char **argv)
{
    cli::Parser parser(argc, argv);
    parser.set_required<std::string>("o", "output", "Output prefix");
	parser.set_optional<int>("n", "number", 10, "Number of output files");
	parser.set_optional<int>("f", "testfraction", 15, "Fraction of events for testing in percent [0-100]");
	parser.set_optional<int>("s", "split", 1, "Number of splits for batched processing");
	parser.set_optional<int>("b", "batch", 0, "Current batch number (<number of splits)");
	parser.set_optional<bool>("t", "truth", true, "Add truth from jetorigin (deactivate for DA)");
    parser.set_required<std::vector<std::string>>("i", "input", "Input files");
    parser.run_and_exit_if_error();
    
    std::string outputPrefix = parser.get<std::string>("o");
    std::cout<<"output file prefix: "<<outputPrefix<<std::endl;
    
    int nOutputs = parser.get<int>("n");
    std::cout<<"output files: "<<nOutputs<<std::endl;
    if (nOutputs<=0)
    {
        std::cout<<"Error: Number of output files (-n) needs to be >=1"<<std::endl;
        return 1;
    }
    
    int nTestFrac = parser.get<int>("f");
    std::cout<<"test fraction: "<<nTestFrac<<"%"<<std::endl;
    if (nTestFrac<0 or nTestFrac>100)
    {
        std::cout<<"Error: Test fraction needs to be within [0;100]"<<std::endl;
        return 1;
    }
    
    int nSplit = parser.get<int>("s");
    std::cout<<"total splits: "<<nSplit<<std::endl;
    if (nSplit<=0)
    {
        std::cout<<"Error: Total split number needs to be >=1!"<<std::endl;
        return 1;
    }
    
    int iSplit = parser.get<int>("b");
    std::cout<<"current split: "<<iSplit<<std::endl;
    if (iSplit>=nSplit)
    {
        std::cout<<"Error: Current split number (-b) needs to be smaller than total split (-s) number!"<<std::endl;
        return 1;
    }
    
    bool addTruth = parser.get<bool>("t");
    std::cout<<"add truth from jetorigin: "<<(addTruth ? "true" : "false")<<std::endl;
    
    std::vector<std::unique_ptr<NanoXTree>> trees;
    std::cout<<"Input files: "<<std::endl;
    
    std::vector<unsigned int> entries;
    unsigned int total_entries = 0;
    
    std::vector<std::vector<std::string>> inputFileNames;
    std::vector<std::vector<std::string>> selectors;
    std::vector<std::vector<std::string>> setters;
    
    std::vector<std::string> inputs = parser.get<std::vector<std::string>>("i");
    if (inputs.size()==0)
    {
        std::cout<<"Error: At least one input file (-i) required!"<<std::endl;
        return 1;
    }
    
    for (const std::string& s: inputs)
    {
        if (ends_with(s,".root"))
        {
            inputFileNames.push_back(std::vector<std::string>{s});
            selectors.push_back(std::vector<std::string>{});
        }
        else if(ends_with(s,".txt"))
        {
            std::ifstream input(s);
            std::vector<std::string> files;
            std::vector<std::string> select;
            std::vector<std::string> setter;
            for( std::string line; getline( input, line ); )
            {
                if (line.size()>0)
                {
                    if (begins_with(line,"#select"))
                    {
                        select.emplace_back(line.begin()+7,line.end());
                    }
                    else if (begins_with(line,"#set"))
                    {
                        setter.emplace_back(line.begin()+4,line.end());
                    }
                    else if (begins_with(line,"#"))
                    {
                        std::cout<<"Ignore unknown instruction: "<<line<<std::endl;
                    }
                    else
                    {
                        files.push_back(line);
                    }
                }
            }
            selectors.push_back(select);
            setters.push_back(setter);
            inputFileNames.push_back(files);
        }
        else
        {
            std::cout<<"Error: Cannot parse file '"<<s<<"'"<<std::endl;
            return 1;
        }
    }
    
    for (size_t i = 0; i < inputFileNames.size(); ++i)
    {
        auto inputFileNameList = inputFileNames[i];
        TChain* chain = new TChain("Events","Events");
        for (const auto& inputFileName: inputFileNameList)
        {
            //std::cout<<"   "<<argv[iarg]<<", nEvents="<<;
            TFile* file = TFile::Open(inputFileName.c_str());
            if (not file)
            {
                std::cout<<"Warning: File '"<<inputFileName<<"' cannot be read"<<std::endl;
                continue;
            }
            
            TTree* tree = dynamic_cast<TTree*>(file->Get("Events"));
            
            if (not tree)
            {
                std::cout<<"Warning: Tree in file '"<<inputFileName<<"' cannot be read"<<std::endl;
                continue;
            }
            int nEvents = tree->GetEntries();
            std::cout<<"   "<<inputFileName<<", nEvents="<<nEvents<<std::endl;
            file->Close();
            chain->AddFile(inputFileName.c_str());
        }
        int nEvents = chain->GetEntries();
        std::cout<<"Total per chain:  "<<nEvents<<std::endl;
        entries.push_back(nEvents);
        total_entries += nEvents;
        trees.emplace_back(std::unique_ptr<NanoXTree>(new NanoXTree (chain,selectors[i],setters[i],addTruth)));
    }
    if (inputFileNames.size()==0)
    {
        std::cout<<"Error: No input files readable!"<<std::endl;
        return 1;
    }
    if (total_entries==0)
    {
        std::cout<<"Error: Total number of entries=0!"<<std::endl;
        return 1;
    }
    
    std::cout<<"Total number of events: "<<total_entries<<std::endl;
    std::vector<std::unique_ptr<UnpackedTree>> unpackedTreesTrain;
    std::vector<std::vector<int>> eventsPerClassPerFileTrain(12,std::vector<int>(nOutputs,0));
    
    std::vector<std::unique_ptr<UnpackedTree>> unpackedTreesTest;
    std::vector<std::vector<int>> eventsPerClassPerFileTest(12,std::vector<int>(nOutputs,0));

    for (unsigned int i = 0; i < nOutputs; ++i)
    {
        unpackedTreesTrain.emplace_back(std::unique_ptr<UnpackedTree>(
            new UnpackedTree(outputPrefix+"_train"+std::to_string(iSplit+1)+"_"+std::to_string(i+1)+".root",addTruth
        )));

        unpackedTreesTest.emplace_back(std::unique_ptr<UnpackedTree>(
            new UnpackedTree(outputPrefix+"_test"+std::to_string(iSplit+1)+"_"+std::to_string(i+1)+".root",addTruth
        )));
    }
    
    
    std::vector<unsigned int> readEvents(entries.size(),0);
    for (unsigned int ientry = 0; ientry<total_entries; ++ientry)
    {
        if (ientry%10000==0)
        {
            std::cout<<"Processing ... "<<100.*ientry/total_entries<<std::endl;
        }
        //use entry number for global splitting; but hash value to split test/train
        if ((ientry%nSplit)!=iSplit)
        {
            continue;
        }
    
        //choose input file pseudo-randomly
        unsigned int hash = ((ientry >> 16) ^ ientry) * 0x45d9f3b;
        hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
        hash = (hash >> 16) ^ hash;
        hash = (hash+hash/total_entries)%total_entries;
        
        unsigned int sum_entries = 0;
        unsigned int ifile = 0;
        for (;ifile<entries.size(); ++ifile)
        {
            sum_entries += entries[ifile];
            if (hash<sum_entries) break;
        }
    
        //ensure that file is not yet at last event; otherwise move to next input file
        while (trees[ifile]->entry()>=entries[ifile])
        {
            ifile=(ifile+1)%trees.size();
        }
        
        //std::cout<<ifile<<": "<<trees[ifile]->entry()<<"/"<<entries[ifile]<<std::endl;
        trees[ifile]->nextEvent();
        readEvents[ifile]+=1;
        
        for (size_t j = 0; j < std::min<int>(8,trees[ifile]->njets()); ++j)
        {
            if (trees[ifile]->isSelected(j))
            {
                int jet_class = trees[ifile]->getJetClass(j);
                
                if (hash%100<nTestFrac)
                {
                    if (jet_class>=0 and jet_class<eventsPerClassPerFileTest.size())
                    {
                        unsigned int ofile = std::distance(
                            eventsPerClassPerFileTest[jet_class].begin(), 
                            std::min_element(
                                eventsPerClassPerFileTest[jet_class].begin(), 
                                eventsPerClassPerFileTest[jet_class].end()
                            )
                        );
                        //std::cout<<ofile<<std::endl;
                        eventsPerClassPerFileTest[jet_class][ofile]+=1;
                        trees[ifile]->unpackJet(j,*unpackedTreesTest[ofile]);
                    }
                }
                else
                {
                    if (jet_class>=0 and jet_class<eventsPerClassPerFileTrain.size())
                    {
                        unsigned int ofile = std::distance(
                            eventsPerClassPerFileTrain[jet_class].begin(), 
                            std::min_element(
                                eventsPerClassPerFileTrain[jet_class].begin(), 
                                eventsPerClassPerFileTrain[jet_class].end()
                            )
                        );
                        //std::cout<<ofile<<std::endl;
                        eventsPerClassPerFileTrain[jet_class][ofile]+=1;
                        trees[ifile]->unpackJet(j,*unpackedTreesTrain[ofile]);
                    }
                }
            }
        }
    }
    
    for (size_t i = 0; i < entries.size(); ++i)
    {
        std::cout<<"infile "<<i<<": found = "<<entries[i]<<", read = "<<readEvents[i]<<std::endl;
    }
    std::cout<<"----- Train ----- "<<std::endl;
    for (size_t c = 0; c < eventsPerClassPerFileTrain.size(); ++c)
    {
        std::cout<<"jet class "<<c<<": ";
        for (size_t i = 0; i < nOutputs; ++i)
        {
            std::cout<<eventsPerClassPerFileTrain[c][i]<<", ";
        }
        std::cout<<std::endl;
    }
    std::cout<<"----- Test ----- "<<std::endl;
    for (size_t c = 0; c < eventsPerClassPerFileTest.size(); ++c)
    {
        std::cout<<"jet class "<<c<<": ";
        for (size_t i = 0; i < nOutputs; ++i)
        {
            std::cout<<eventsPerClassPerFileTest[c][i]<<", ";
        }
        std::cout<<std::endl;
    }
    
    for (auto& unpackedTree: unpackedTreesTrain)
    {
        unpackedTree->close();
    }
    
    for (auto& unpackedTree: unpackedTreesTest)
    {
        unpackedTree->close();
    }
    
    return 0;
}
