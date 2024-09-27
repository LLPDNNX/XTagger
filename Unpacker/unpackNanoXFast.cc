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
        
        
        static constexpr int maxEntries_cpf = 25;
        static constexpr int maxEntries_npf = 25;
        static constexpr int maxEntries_sv = 4;
        static constexpr int maxEntries_muon = 2;
        static constexpr int maxEntries_electron = 2;
        
        static constexpr int bufferSize = 64000; //default is 32kB
        
        float jetorigin_isPU;
        float jetorigin_isB;
        float jetorigin_isBB;
        float jetorigin_isGBB;
        float jetorigin_isLeptonic_B;
        float jetorigin_isLeptonic_C;
        float jetorigin_isC;
        float jetorigin_isCC;
        float jetorigin_isGCC;
        float jetorigin_isS;
        float jetorigin_isUD;
        float jetorigin_isG;
        //Include LLP flavour : 
        float jetorigin_isLLP_RAD; //no flavour match (likely from wide angle radiation)
        float jetorigin_isLLP_MU; //prompt lepton
        float jetorigin_isLLP_E; //prompt lepton
        float jetorigin_isLLP_Q; //single light quark
        float jetorigin_isLLP_QMU; //single light quark + prompt lepton
        float jetorigin_isLLP_QE; //single light quark + prompt lepton
        float jetorigin_isLLP_QQ; //double light quark
        float jetorigin_isLLP_QQMU; //double light quark + prompt lepton
        float jetorigin_isLLP_QQE; //double light quark + prompt lepton
        float jetorigin_isLLP_B; //single b/c quark
        float jetorigin_isLLP_BMU; //single b/c quark + prompt lepton
        float jetorigin_isLLP_BE; //single b/c quark + prompt lepton
        float jetorigin_isLLP_BB; //double b/c quark
        float jetorigin_isLLP_BBMU; //double b/c quark + prompt lepton
        float jetorigin_isLLP_BBE; //double b/c quark + prompt lepton
        float jetorigin_isUndefined;
        
        float jetorigin_displacement;
        float jetorigin_decay_angle;
        float jetorigin_displacement_xy;
        float jetorigin_displacement_z ; 
        float jetorigin_ctau;
        float jetorigin_betagamma;
        float jetorigin_partonFlavor;
        float jetorigin_hadronFlavor;
        float jetorigin_llpId;
        float jetorigin_llp_mass;
        float jetorigin_llp_pt;

        //unsigned int nglobal;
        float global_pt;
        float global_eta;
        float global_mass;
        float global_n60;
        float global_n90;
        float global_chargedEmEnergyFraction;
        float global_chargedHadronEnergyFraction;
        float global_chargedMuEnergyFraction;
        float global_electronEnergyFraction;

        float global_tau1;
        float global_tau2;
        float global_tau3;
    
        float global_relMassDropMassAK;
        float global_relMassDropMassCA;
        float global_relSoftDropMassAK;
        float global_relSoftDropMassCA;
    
        float global_thrust;
        float global_sphericity;
        float global_circularity;
        float global_isotropy;
        float global_eventShapeC;
        float global_eventShapeD;
        
        float csv_trackSumJetEtRatio;
        float csv_trackSumJetDeltaR;
        float csv_vertexCategory;
        float csv_trackSip2dValAboveCharm;
        float csv_trackSip2dSigAboveCharm;
        float csv_trackSip3dValAboveCharm;
        float csv_trackSip3dSigAboveCharm;
        float csv_jetNSelectedTracks;
        float csv_jetNTracksEtaRel;

        float isData;
        float xsecweight;
        float processId;
        
        unsigned int ncpf;
        float cpf_trackEtaRel[maxEntries_cpf];
        float cpf_trackPtRel[maxEntries_cpf];
        float cpf_trackPPar[maxEntries_cpf];
        float cpf_trackDeltaR[maxEntries_cpf];
        float cpf_trackPtRatio[maxEntries_cpf];
        float cpf_trackPParRatio[maxEntries_cpf];
        float cpf_trackSip2dVal[maxEntries_cpf];
        float cpf_trackSip2dSig[maxEntries_cpf];
        float cpf_trackSip3dVal[maxEntries_cpf];
        float cpf_trackSip3dSig[maxEntries_cpf];
        float cpf_trackJetDistVal[maxEntries_cpf];
        float cpf_trackJetDistSig[maxEntries_cpf];
        float cpf_ptrel[maxEntries_cpf];
        float cpf_deta[maxEntries_cpf];
        float cpf_dphi[maxEntries_cpf];
        float cpf_drminsv[maxEntries_cpf];
        float cpf_vertex_association[maxEntries_cpf];
        float cpf_fromPV[maxEntries_cpf];
        float cpf_puppi_weight[maxEntries_cpf];
        float cpf_track_chi2[maxEntries_cpf];
        float cpf_track_ndof[maxEntries_cpf];
        float cpf_track_quality[maxEntries_cpf];
        float cpf_relmassdrop[maxEntries_cpf];
        
        float cpf_matchedSV[maxEntries_cpf];
        float cpf_matchedMuon[maxEntries_cpf];
        float cpf_matchedElectron[maxEntries_cpf];

        float cpf_track_numberOfValidPixelHits [maxEntries_cpf] ; 
 	float cpf_track_pixelLayersWithMeasurement[maxEntries_cpf];
	float cpf_track_numberOfValidStripHits[maxEntries_cpf];
        float cpf_track_stripLayersWithMeasurement[maxEntries_cpf];
        
        unsigned int nnpf;
        float npf_ptrel[maxEntries_npf];
        float npf_deta[maxEntries_npf];
        float npf_dphi[maxEntries_npf];
        float npf_deltaR[maxEntries_npf];
        float npf_isGamma[maxEntries_npf];
        float npf_hcal_fraction[maxEntries_npf];
        float npf_drminsv[maxEntries_npf];
        float npf_puppi_weight[maxEntries_npf];
        float npf_relmassdrop[maxEntries_npf];
        
        unsigned int nsv;
        float sv_ptrel[maxEntries_sv];
        float sv_deta[maxEntries_sv];
        float sv_dphi[maxEntries_sv];
        float sv_mass[maxEntries_sv];
        float sv_deltaR[maxEntries_sv];
        float sv_ntracks[maxEntries_sv];
        float sv_chi2[maxEntries_sv];
        float sv_ndof[maxEntries_sv];
        float sv_dxy[maxEntries_sv];
        float sv_dxysig[maxEntries_sv];
        float sv_d3d[maxEntries_sv];
        float sv_d3dsig[maxEntries_sv];
        float sv_costhetasvpv[maxEntries_sv];
        float sv_enratio[maxEntries_sv];

        unsigned int nmuon;
        float muon_isGlobal[maxEntries_muon] ; 
        float muon_isTight[maxEntries_muon] ; 
        float muon_isMedium[maxEntries_muon] ; 
        float muon_isLoose[maxEntries_muon] ; 
        float muon_isStandAlone[maxEntries_muon] ;

        float muon_ptrel [maxEntries_muon];
        float muon_EtaRel[maxEntries_muon];
        float muon_dphi[maxEntries_muon];
        float muon_deta[maxEntries_muon];
        float muon_charge [maxEntries_muon]; 
        float muon_energy[maxEntries_muon];
        float muon_jetDeltaR [maxEntries_muon]; 
        float muon_numberOfMatchedStations [maxEntries_muon];

        float muon_2dIp [maxEntries_muon]; 
        float muon_2dIpSig [maxEntries_muon];
        float muon_3dIp [maxEntries_muon]; 
        float muon_3dIpSig [maxEntries_muon]; 

        float muon_dxy [maxEntries_muon]; 
        float muon_dxyError [maxEntries_muon]; 
        float muon_dxySig [maxEntries_muon]; 
        float muon_dz [maxEntries_muon]; 
        float muon_dzError [maxEntries_muon]; 
        float muon_numberOfValidPixelHits[maxEntries_muon]; 
        float muon_numberOfpixelLayersWithMeasurement [maxEntries_muon]; 

        float muon_chi2 [maxEntries_muon]; 
        float muon_ndof [maxEntries_muon]; 

        float muon_caloIso [maxEntries_muon]; 
        float muon_ecalIso [maxEntries_muon]; 
        float muon_hcalIso [maxEntries_muon];

        float muon_sumPfChHadronPt [maxEntries_muon]; 
        float muon_sumPfNeuHadronEt [maxEntries_muon]; 
        float muon_Pfpileup [maxEntries_muon]; 
        float muon_sumPfPhotonEt [maxEntries_muon]; 

        float muon_sumPfChHadronPt03 [maxEntries_muon]; 
        float muon_sumPfNeuHadronEt03 [maxEntries_muon]; 
        float muon_Pfpileup03 [maxEntries_muon]; 
        float muon_sumPfPhotonEt03 [maxEntries_muon]; 

        float muon_sumChHadronPt [maxEntries_muon]; 
        float muon_sumNeuHadronEt [maxEntries_muon]; 
        float muon_pileup [maxEntries_muon]; 
        float muon_sumPhotonEt [maxEntries_muon]; 

        float muon_timeAtIpInOut [maxEntries_muon]; 
        float muon_timeAtIpInOutErr [maxEntries_muon]; 
        float muon_timeAtIpOutIn [maxEntries_muon];

        unsigned int nelectron;
	int electron_jetIdx[maxEntries_electron];
        float electron_ptrel[maxEntries_electron];
        float electron_jetDeltaR[maxEntries_electron]; 
        float electron_deta[maxEntries_electron];
        float electron_dphi[maxEntries_electron];
        float electron_charge[maxEntries_electron]; 
        float electron_energy[maxEntries_electron];
        float electron_EtFromCaloEn[maxEntries_electron];
        float electron_isEB[maxEntries_electron]; 
        float electron_isEE[maxEntries_electron]; 
        float electron_ecalEnergy[maxEntries_electron]; 
        float electron_isPassConversionVeto[maxEntries_electron];
        float electron_convDist[maxEntries_electron]; 
        int   electron_convFlags[maxEntries_electron]; 
        float electron_convRadius[maxEntries_electron]; 
        float electron_hadronicOverEm[maxEntries_electron];
        float electron_ecalDrivenSeed[maxEntries_electron];

    // superCluster block

        float electron_SC_energy[maxEntries_electron]; 
        float electron_SC_deta[maxEntries_electron]; 
        float electron_SC_dphi[maxEntries_electron];
        float electron_SC_et[maxEntries_electron];
        float electron_SC_eSuperClusterOverP[maxEntries_electron]; 
        float electron_scE1x5Overe5x5[maxEntries_electron]; 
        float electron_scE2x5MaxOvere5x5[maxEntries_electron]; 
        float electron_scE5x5[maxEntries_electron]; 
        float electron_scE5x5Rel[maxEntries_electron]; 
        float electron_scPixCharge[maxEntries_electron]; 
        float electron_scSigmaEtaEta[maxEntries_electron];
        float electron_scSigmaIEtaIEta[maxEntries_electron];  
        float electron_superClusterFbrem[maxEntries_electron]; 

        float electron_2dIP[maxEntries_electron]; 
        float electron_2dIPSig[maxEntries_electron];
        float electron_3dIP[maxEntries_electron]; 
        float electron_3dIPSig[maxEntries_electron]; 
        float electron_eSeedClusterOverP[maxEntries_electron];
        float electron_eSeedClusterOverPout[maxEntries_electron];
        float electron_eSuperClusterOverP[maxEntries_electron];
        float electron_eTopOvere5x5[maxEntries_electron]; 

        float electron_deltaEtaEleClusterTrackAtCalo[maxEntries_electron]; 
        float electron_deltaEtaSeedClusterTrackAtCalo[maxEntries_electron];
        float electron_deltaPhiSeedClusterTrackAtCalo[maxEntries_electron]; 
        float electron_deltaEtaSeedClusterTrackAtVtx[maxEntries_electron]; 
        float electron_deltaEtaSuperClusterTrackAtVtx[maxEntries_electron];
        float electron_deltaPhiEleClusterTrackAtCalo[maxEntries_electron]; 
        float electron_deltaPhiSuperClusterTrackAtVtx[maxEntries_electron];
        float electron_sCseedEta[maxEntries_electron];  

    // electron gsf variables. 
        float electron_EtaRel[maxEntries_electron]; 
        float electron_dxy[maxEntries_electron]; 
        float electron_dz[maxEntries_electron];
        float electron_nbOfMissingHits[maxEntries_electron]; 
        float electron_gsfCharge[maxEntries_electron];


        int   electron_numberOfBrems[maxEntries_electron];
        float electron_trackFbrem[maxEntries_electron]; 
        float electron_fbrem[maxEntries_electron]; 
        float electron_e5x5[maxEntries_electron]; 
        float electron_e5x5Rel[maxEntries_electron]; 
        float electron_e1x5Overe5x5[maxEntries_electron]; 
        float electron_e2x5MaxOvere5x5[maxEntries_electron];

    // 5*5 cells cluster 
    //

        float electron_full5x5_e5x5[maxEntries_electron];
        float electron_full5x5_e5x5Rel[maxEntries_electron]; 
        float electron_full5x5_sigmaIetaIeta[maxEntries_electron];
        float electron_full5x5_e1x5Overe5x5[maxEntries_electron];
        float electron_full5x5_e2x5BottomOvere5x5[maxEntries_electron];
        float electron_full5x5_e2x5LeftOvere5x5[maxEntries_electron];
        float electron_full5x5_e2x5MaxOvere5x5[maxEntries_electron];
        float electron_full5x5_e2x5RightOvere5x5[maxEntries_electron];
        float electron_full5x5_e2x5TopOvere5x5[maxEntries_electron];



        float electron_full5x5_eBottomOvere5x5[maxEntries_electron];
        float electron_full5x5_eLeftOvere5x5[maxEntries_electron];
        float electron_full5x5_eRightOvere5x5[maxEntries_electron];
        float electron_full5x5_eTopOvere5x5[maxEntries_electron];
        float electron_full5x5_hcalDepth1OverEcal[maxEntries_electron];
        float electron_full5x5_hcalDepth1OverEcalBc[maxEntries_electron];
        float electron_full5x5_hcalDepth2OverEcal[maxEntries_electron];
        float electron_full5x5_hcalDepth2OverEcalBc[maxEntries_electron];
        float electron_full5x5_hcalOverEcal[maxEntries_electron];
        float electron_full5x5_hcalOverEcalBc[maxEntries_electron];   
        float electron_full5x5_r9[maxEntries_electron];



    // Isolation block
        float electron_neutralHadronIso[maxEntries_electron]; 
        float electron_particleIso [maxEntries_electron]; 
        float electron_photonIso[maxEntries_electron];
        float electron_puChargedHadronIso[maxEntries_electron]; 
        float electron_trackIso[maxEntries_electron];  
        float electron_hcalDepth1OverEcal[maxEntries_electron]; 
        float electron_hcalDepth2OverEcal[maxEntries_electron]; 
        float electron_ecalPFClusterIso[maxEntries_electron];
        float electron_hcalPFClusterIso[maxEntries_electron];  
        float electron_dr03TkSumPt[maxEntries_electron]; 

        float electron_dr03EcalRecHitSumEt[maxEntries_electron]; 
        float electron_dr03HcalDepth1TowerSumEt[maxEntries_electron];  
        float electron_dr03HcalDepth1TowerSumEtBc[maxEntries_electron]; 
        float electron_dr03HcalDepth2TowerSumEt[maxEntries_electron]; 
        float electron_dr03HcalDepth2TowerSumEtBc[maxEntries_electron]; 
        float electron_pfSumPhotonEt[maxEntries_electron]; 
        float electron_pfSumChargedHadronPt[maxEntries_electron]; 
        float electron_pfSumNeutralHadronEt[maxEntries_electron]; 
        float electron_pfSumPUPt[maxEntries_electron];

        float electron_dr04EcalRecHitSumEt[maxEntries_electron];  
        float electron_dr04HcalDepth1TowerSumEt[maxEntries_electron];  
        float electron_dr04HcalDepth1TowerSumEtBc[maxEntries_electron];
        float electron_dr04HcalDepth2TowerSumEt[maxEntries_electron]; 
        float electron_dr04HcalDepth2TowerSumEtBc [maxEntries_electron];
        float electron_dr04HcalTowerSumEt[maxEntries_electron];
        float electron_dr04HcalTowerSumEtBc[maxEntries_electron];
	

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
                tree_->Branch("jetorigin_isPU",&jetorigin_isPU,"jetorigin_isPU/F",bufferSize);
                tree_->Branch("jetorigin_isB",&jetorigin_isB,"jetorigin_isB/F",bufferSize);
                tree_->Branch("jetorigin_isBB",&jetorigin_isBB,"jetorigin_isBB/F",bufferSize);
                tree_->Branch("jetorigin_isGBB",&jetorigin_isGBB,"jetorigin_isGBB/F",bufferSize);
                tree_->Branch("jetorigin_isLeptonic_B",&jetorigin_isLeptonic_B,"jetorigin_isLeptonic_B/F",bufferSize);
                tree_->Branch("jetorigin_isLeptonic_C",&jetorigin_isLeptonic_C,"jetorigin_isLeptonic_C/F",bufferSize);
                tree_->Branch("jetorigin_isC",&jetorigin_isC,"jetorigin_isC/F",bufferSize);
                tree_->Branch("jetorigin_isCC",&jetorigin_isCC,"jetorigin_isCC/F",bufferSize);
                tree_->Branch("jetorigin_isGCC",&jetorigin_isGCC,"jetorigin_isGCC/F",bufferSize);
                tree_->Branch("jetorigin_isS",&jetorigin_isS,"jetorigin_isS/F",bufferSize);
                tree_->Branch("jetorigin_isUD",&jetorigin_isUD,"jetorigin_isUD/F",bufferSize);
                tree_->Branch("jetorigin_isG",&jetorigin_isG,"jetorigin_isG/F",bufferSize);
                
                //Add LLP flavour : 
                tree_->Branch("jetorigin_isLLP_RAD",&jetorigin_isLLP_RAD , "jetorigin_isLLP_RAD/F", bufferSize) ; 
                tree_->Branch("jetorigin_isLLP_MU",&jetorigin_isLLP_MU , "jetorigin_isLLP_MU/F", bufferSize) ; 
                tree_->Branch("jetorigin_isLLP_E",&jetorigin_isLLP_E , "jetorigin_isLLP_E/F", bufferSize) ;
                tree_->Branch("jetorigin_isLLP_Q",&jetorigin_isLLP_Q , "jetorigin_isLLP_Q/F", bufferSize) ; 
                tree_->Branch("jetorigin_isLLP_QMU",&jetorigin_isLLP_QMU, "jetorigin_isLLP_QMU/F", bufferSize) ; 
                tree_->Branch("jetorigin_isLLP_QE",&jetorigin_isLLP_QE , "jetorigin_isLLP_QE/F", bufferSize) ; 
                tree_->Branch("jetorigin_isLLP_QQ",&jetorigin_isLLP_QQ , "jetorigin_isLLP_QQ/F", bufferSize) ; 
                tree_->Branch("jetorigin_isLLP_QQMU",&jetorigin_isLLP_QQMU ,"jetorigin_isLLP_QQMU/F", bufferSize) ; 
                tree_->Branch("jetorigin_isLLP_QQE",&jetorigin_isLLP_QQE , "jetorigin_isLLP_QQE/F", bufferSize) ; 
                tree_->Branch("jetorigin_isLLP_QQE",&jetorigin_isLLP_QQE ,"jetorigin_isLLP_QQE/F", bufferSize) ; 
                tree_->Branch("jetorigin_isLLP_BMU",&jetorigin_isLLP_BMU , "jetorigin_isLLP_BMU/F", bufferSize) ; 
                tree_->Branch("jetorigin_isLLP_BE",&jetorigin_isLLP_BE ,"jetorigin_isLLP_BE/F", bufferSize) ; 
                tree_->Branch("jetorigin_isLLP_BB",&jetorigin_isLLP_BB, "jetorigin_isLLP_BB/F", bufferSize) ; 
                tree_->Branch("jetorigin_isLLP_BBMU",&jetorigin_isLLP_BBMU , "jetorigin_isLLP_BBMU/F", bufferSize) ; 
                tree_->Branch("jetorigin_isLLP_BBE",&jetorigin_isLLP_BBE , "jetorigin_isLLP_BBE/F", bufferSize) ; 

                tree_->Branch("jetorigin_isUndefined",&jetorigin_isUndefined,"jetorigin_isUndefined/F",bufferSize);
                
                tree_->Branch("jetorigin_displacement",&jetorigin_displacement,"jetorigin_displacement/F",bufferSize);
                tree_->Branch("jetorigin_decay_angle",&jetorigin_decay_angle,"jetorigin_decay_angle/F",bufferSize);
                tree_->Branch("jetorigin_displacement_xy" , &jetorigin_displacement_xy ,"jetorigin_displacement_xy/F" , bufferSize); 
                tree_->Branch("jetorigin_displacement_z" , &jetorigin_displacement_z ,"jetorigin_displacement_z/F" ,bufferSize); 
                tree_->Branch("jetorigin_ctau" , &jetorigin_ctau ,"jetorigin_ctau/F" ,bufferSize); 
                tree_->Branch("jetorigin_betagamma", &jetorigin_betagamma ,"jetorigin_betagamma/F" ,bufferSize); 
                tree_->Branch("jetorigin_partonFlavor", &jetorigin_partonFlavor , "jetorigin_partonFlavor/F" , bufferSize ) ;
                tree_->Branch("jetorigin_hadronFlavor", &jetorigin_hadronFlavor , "jetorigin_hadronFlavor/F" , bufferSize ) ;
                tree_->Branch("jetorigin_llpId", &jetorigin_llpId , "jetorigin_llpId/F" , bufferSize ) ;
                tree_->Branch("jetorigin_llp_mass", &jetorigin_llp_mass , "jetorigin_llp_mass/F" , bufferSize ) ;
                tree_->Branch("jetorigin_llp_pt", &jetorigin_llp_pt , "jetorigin_llp_pt/F" , bufferSize ) ;
            }
            else
            {
                tree_->Branch("xsecweight",&xsecweight,"xsecweight/F",bufferSize);
                tree_->Branch("isData",&isData,"isData/F",bufferSize);
                tree_->Branch("processId",&processId,"processId/F",bufferSize);
            }
            
            tree_->Branch("global_pt",&global_pt,"global_pt/F",bufferSize);
            tree_->Branch("global_eta",&global_eta,"global_eta/F",bufferSize);

            tree_->Branch("global_mass",&global_mass,"global_mass/F", bufferSize);
            tree_->Branch("global_n60",&global_n60,"global_n60/F", bufferSize);
            tree_->Branch("global_n90",&global_n90,"global_n90/F", bufferSize);
            tree_->Branch("global_chargedEmEnergyFraction",&global_chargedEmEnergyFraction,"global_chargedEmEnergyFraction/F", bufferSize);
            tree_->Branch("global_chargedHadronEnergyFraction",&global_chargedHadronEnergyFraction,"global_chargedHadronEnergyFraction/F", bufferSize);
            tree_->Branch("global_chargedMuEnergyFraction",&global_chargedMuEnergyFraction,"global_chargedMuEnergyFraction/F", bufferSize);
            tree_->Branch("global_electronEnergyFraction",&global_electronEnergyFraction,"global_electronEnergyFraction/F", bufferSize);

            tree_->Branch("global_tau1",&global_tau1,"global_tau1/F", bufferSize);
            tree_->Branch("global_tau2",&global_tau2,"global_tau2/F", bufferSize);
            tree_->Branch("global_tau3",&global_tau3,"global_tau3/F", bufferSize);
    
            tree_->Branch("global_relMassDropMassAK",&global_relMassDropMassAK,"global_relMassDropMassAK/F", bufferSize);
            tree_->Branch("global_relMassDropMassCA",&global_relMassDropMassCA,"global_relMassDropMassCA/F", bufferSize);
            tree_->Branch("global_relSoftDropMassAK",&global_relSoftDropMassAK,"global_relSoftDropMassAK/F", bufferSize);
            tree_->Branch("global_relSoftDropMassCA",&global_relSoftDropMassCA,"global_relSoftDropMassCA/F", bufferSize);
    
            tree_->Branch("global_thrust",&global_thrust,"global_thrust/F", bufferSize);
            tree_->Branch("global_sphericity",&global_sphericity,"global_sphericity/F", bufferSize);
            tree_->Branch("global_circularity",&global_circularity,"global_circularity/F", bufferSize);
            tree_->Branch("global_isotropy",&global_isotropy,"global_isotropy/F", bufferSize);
            tree_->Branch("global_eventShapeC",&global_eventShapeC,"global_eventShapeC/F", bufferSize);
            tree_->Branch("global_eventShapeD",&global_eventShapeD ,"global_eventShapeD/F", bufferSize);
            
            
            tree_->Branch("csv_trackSumJetEtRatio",&csv_trackSumJetEtRatio,"csv_trackSumJetEtRatio/F",bufferSize);
            tree_->Branch("csv_trackSumJetDeltaR",&csv_trackSumJetDeltaR,"csv_trackSumJetDeltaR/F",bufferSize);
            tree_->Branch("csv_vertexCategory",&csv_vertexCategory,"csv_vertexCategory/F",bufferSize);
            tree_->Branch("csv_trackSip2dValAboveCharm",&csv_trackSip2dValAboveCharm,"csv_trackSip2dValAboveCharm/F",bufferSize);
            tree_->Branch("csv_trackSip2dSigAboveCharm",&csv_trackSip2dSigAboveCharm,"csv_trackSip2dSigAboveCharm/F",bufferSize);
            tree_->Branch("csv_trackSip3dValAboveCharm",&csv_trackSip3dValAboveCharm,"csv_trackSip3dValAboveCharm/F",bufferSize);
            tree_->Branch("csv_trackSip3dSigAboveCharm",&csv_trackSip3dSigAboveCharm,"csv_trackSip3dSigAboveCharm/F",bufferSize);
            tree_->Branch("csv_jetNSelectedTracks",&csv_jetNSelectedTracks,"csv_jetNSelectedTracks/F",bufferSize);
            tree_->Branch("csv_jetNTracksEtaRel",&csv_jetNTracksEtaRel,"csv_jetNTracksEtaRel/F",bufferSize);
            

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
            tree_->Branch("cpf_deta",&cpf_deta,"cpf_deta[ncpf]/F",bufferSize);
            tree_->Branch("cpf_dphi",&cpf_dphi,"cpf_dphi[ncpf]/F",bufferSize);
            tree_->Branch("cpf_drminsv",&cpf_drminsv,"cpf_drminsv[ncpf]/F",bufferSize);
            tree_->Branch("cpf_vertex_association",&cpf_vertex_association,"cpf_vertex_association[ncpf]/F",bufferSize);
            tree_->Branch("cpf_fromPV",&cpf_fromPV,"cpf_fromPV[ncpf]/F",bufferSize);
            tree_->Branch("cpf_puppi_weight",&cpf_puppi_weight,"cpf_puppi_weight[ncpf]/F",bufferSize);
            tree_->Branch("cpf_track_chi2",&cpf_track_chi2,"cpf_track_chi2[ncpf]/F",bufferSize);
            tree_->Branch("cpf_track_ndof",&cpf_track_ndof,"cpf_track_ndof[ncpf]/F",bufferSize);
            tree_->Branch("cpf_track_quality",&cpf_track_quality,"cpf_track_quality[ncpf]/F",bufferSize);
            tree_->Branch("cpf_relmassdrop",&cpf_relmassdrop,"cpf_relmassdrop[ncpf]/F",bufferSize);
            
            tree_->Branch("cpf_matchedSV",&cpf_matchedSV,"cpf_matchedSV[ncpf]/F",bufferSize);
            tree_->Branch("cpf_matchedMuon",&cpf_matchedMuon,"cpf_matchedMuon[ncpf]/F",bufferSize);
            tree_->Branch("cpf_matchedElectron",&cpf_matchedElectron,"cpf_matchedElectron[ncpf]/F",bufferSize);
	    tree_->Branch("cpf_track_numberOfValidPixelHits", &cpf_track_numberOfValidPixelHits , "cpf_track_numberOfValidPixelHits[ncpf]/F", bufferSize) ; 
	    tree_->Branch("cpf_track_pixelLayersWithMeasurement", &cpf_track_pixelLayersWithMeasurement, " cpf_track_pixelLayersWithMeasurement[ncpf]/F", bufferSize); 
	    tree_->Branch("cpf_track_numberOfValidStripHits", &cpf_track_numberOfValidStripHits ,"cpf_track_numberOfValidStripHits[ncpf]/F", bufferSize); 
	    tree_->Branch("cpf_track_stripLayersWithMeasurement", &cpf_track_stripLayersWithMeasurement, "cpf_track_stripLayersWithMeasurement[ncpf]/F", bufferSize);  
            

            tree_->Branch("nnpf",&nnpf,"nnpf/I",bufferSize);
            tree_->Branch("npf_ptrel",&npf_ptrel,"npf_ptrel[nnpf]/F",bufferSize);
            tree_->Branch("npf_deta",&npf_deta,"npf_deta[nnpf]/F",bufferSize);
            tree_->Branch("npf_dphi",&npf_dphi,"npf_dphi[nnpf]/F",bufferSize);
            tree_->Branch("npf_deltaR",&npf_deltaR,"npf_deltaR[nnpf]/F",bufferSize);
            tree_->Branch("npf_isGamma",&npf_isGamma,"npf_isGamma[nnpf]/F",bufferSize);
            tree_->Branch("npf_hcal_fraction",&npf_hcal_fraction,"npf_hcal_fraction[nnpf]/F",bufferSize);
            tree_->Branch("npf_drminsv",&npf_drminsv,"npf_drminsv[nnpf]/F",bufferSize);
            tree_->Branch("npf_puppi_weight",&npf_puppi_weight,"npf_puppi_weight[nnpf]/F",bufferSize);
            tree_->Branch("npf_relmassdrop",&npf_relmassdrop,"npf_relmassdrop[nnpf]/F",bufferSize);

            tree_->Branch("nsv",&nsv,"nsv/I",bufferSize);
            tree_->Branch("sv_ptrel",&sv_ptrel,"sv_ptrel[nsv]/F",bufferSize);
            tree_->Branch("sv_deta",&sv_deta,"sv_deta[nsv]/F",bufferSize);
            tree_->Branch("sv_dphi",&sv_dphi,"sv_dphi[nsv]/F",bufferSize);
            tree_->Branch("sv_mass",&sv_deltaR,"sv_mass[nsv]/F",bufferSize);
            tree_->Branch("sv_deltaR",&sv_deltaR,"sv_deltaR[nsv]/F",bufferSize);
            tree_->Branch("sv_ntracks",&sv_ntracks,"sv_ntracks[nsv]/F",bufferSize);
            tree_->Branch("sv_chi2",&sv_chi2,"sv_chi2[nsv]/F",bufferSize);
            tree_->Branch("sv_ndof",&sv_ndof,"sv_ndof[nsv]/F",bufferSize);
            tree_->Branch("sv_dxy",&sv_dxy,"sv_dxy[nsv]/F",bufferSize);
            tree_->Branch("sv_dxysig",&sv_dxysig,"sv_dxysig[nsv]/F",bufferSize);
            tree_->Branch("sv_d3d",&sv_d3d,"sv_d3d[nsv]/F",bufferSize);
            tree_->Branch("sv_d3dsig",&sv_d3dsig,"sv_d3dsig[nsv]/F",bufferSize);
            tree_->Branch("sv_costhetasvpv",&sv_costhetasvpv,"sv_costhetasvpv[nsv]/F",bufferSize);
            tree_->Branch("sv_enratio",&sv_enratio,"sv_enratio[nsv]/F",bufferSize);

		
    	    tree_->Branch("nmuon",&nmuon,"nmuon/I",bufferSize); 
            tree_->Branch("muon_isGlobal",&muon_isGlobal, "muon_isGlobal[nmuon]/F",bufferSize); 
            tree_->Branch("muon_isTight",&muon_isTight,"muon_isTight[nmuon]/F",bufferSize); 
            tree_->Branch("muon_isMedium",&muon_isMedium,"muon_isMedium[nmuon]/F",bufferSize); 
            tree_->Branch("muon_isLoose",&muon_isLoose,"muon_isLoose[nmuon]/F",bufferSize); 
            tree_->Branch("muon_isStandAlone",&muon_isStandAlone,"muon_isStandAlone[nmuon]/F",bufferSize);

    
            tree_->Branch("muon_ptrel", &muon_ptrel,"muon_ptrel[nmuon]/F",bufferSize);
            tree_->Branch("muon_EtaRel", &muon_EtaRel,"muon_EtaRel[nmuon]/F",bufferSize);
            tree_->Branch("muon_dphi",&muon_dphi,"muon_dphi[nmuon]/F",bufferSize);
            tree_->Branch("muon_deta",&muon_dphi,"muon_deta[nmuon]/F",bufferSize);
            tree_->Branch("muon_charge",&muon_charge,"muon_charge[nmuon]/F",bufferSize); 
            tree_->Branch("muon_energy",&muon_energy,"muon_energy[nmuon]/F",bufferSize);
            tree_->Branch("muon_jetDeltaR",&muon_jetDeltaR,"muon_jetDeltaR[nmuon]/F",bufferSize); 
            tree_->Branch("muon_numberOfMatchedStations",&muon_numberOfMatchedStations,"muon_numberOfMatchedStations[nmuon]/F",bufferSize);

            tree_->Branch("muon_2dIp", &muon_2dIp,"muon_2dIp[nmuon]/F",bufferSize); 
            tree_->Branch("muon_2dIpSig", &muon_2dIpSig, "muon_2dIpSi[nmuon]/F",bufferSize);
            tree_->Branch("muon_3dIp",&muon_3dIp,"muon_3dIp[nmuon]/F",bufferSize); 
            tree_->Branch("muon_3dIpSig",&muon_3dIpSig,"muon_3dIpSig[nmuon]/F",bufferSize); 

            tree_->Branch("muon_dxy", &muon_dxy, "muon_dxy[nmuon]/F",bufferSize); 
            tree_->Branch("muon_dxyError", &muon_dxyError,"muon_dxyError[nmuon]/F",bufferSize); 
            tree_->Branch("muon_dxySig",&muon_dxySig,"muon_dxySig[nmuon]/F",bufferSize); 
            tree_->Branch("muon_dz",&muon_dz, "muon_dz[nmuon]/F",bufferSize); 
            tree_->Branch("muon_dzError", &muon_dzError,"muon_dzError[nmuon]/F",bufferSize); 
            tree_->Branch("muon_numberOfValidPixelHits",&muon_numberOfValidPixelHits, "muon_numberOfValidPixelHits[nmuon]/F",bufferSize); 
            tree_->Branch("muon_numberOfpixelLayersWithMeasurement", &muon_numberOfpixelLayersWithMeasurement, "muon_numberOfpixelLayersWithMeasurement[nmuon]/F",bufferSize); 


            tree_->Branch("muon_chi2", &muon_chi2, "muon_chi2[nmuon]/F",bufferSize); 
            tree_->Branch("muon_ndof",&muon_ndof, "muon_ndof[nmuon]/F",bufferSize); 

            tree_->Branch("muon_caloIso",&muon_caloIso,"muon_caloIso[nmuon]/F",bufferSize); 
            tree_->Branch("muon_ecalIso",&muon_ecalIso,"muon_ecalIso[nmuon]/F",bufferSize); 
            tree_->Branch("muon_hcalIso",&muon_hcalIso,"muon_hcalIso[nmuon]/F",bufferSize);

            tree_->Branch("muon_sumPfChHadronPt",&muon_sumPfChHadronPt,"muon_sumPfChHadronPt[nmuon]/F",bufferSize); 
            tree_->Branch("muon_sumPfNeuHadronEt",&muon_sumPfNeuHadronEt,"muon_sumPfNeuHadronEt[nmuon]/F",bufferSize); 
            tree_->Branch("muon_Pfpileup",&muon_Pfpileup,"muon_Pfpileup[nmuon]/F",bufferSize); 
            tree_->Branch("muon_sumPfPhotonEt",&muon_sumPfPhotonEt,"muon_sumPfPhotonEt[nmuon]/F",bufferSize); 

            tree_->Branch("muon_sumPfChHadronPt03",&muon_sumPfChHadronPt03, "muon_sumPfChHadronPt03[nmuon]/F",bufferSize); 
            tree_->Branch("muon_sumPfNeuHadronEt03",&muon_sumPfNeuHadronEt03,"muon_sumPfNeuHadronEt03[nmuon]/F",bufferSize); 
            tree_->Branch("muon_Pfpileup03",&muon_Pfpileup03,"muon_Pfpileup03[nmuon]/F",bufferSize); 
            tree_->Branch("muon_sumPfPhotonEt03",&muon_sumPfPhotonEt03,"muon_sumPfPhotonEt03[nmuon]/F",bufferSize); 

            tree_->Branch("muon_sumChHadronPt",&muon_sumChHadronPt, "muon_sumChHadronPt[nmuon]/F",bufferSize);

	    
            tree_->Branch("nelectron",&nelectron,"nelectron/I",bufferSize);
            tree_->Branch("electron_jetIdx",&electron_jetIdx,"electron_jetIdx/I",bufferSize);
            tree_->Branch("electron_ptrel",&electron_ptrel,"electron_ptrel[nelectron]/F",bufferSize);
            tree_->Branch("electron_jetDeltaR",&electron_jetDeltaR,"electron_jetDeltaR[nelectron]/F" ,bufferSize); 
            tree_->Branch("electron_deta",&electron_deta,"electron_deta[nelectron]/F",bufferSize);
            tree_->Branch("electron_dphi",&electron_dphi,"electron_dphi[nelectron]/F",bufferSize);
            tree_->Branch("electron_charge",&electron_charge,"electron_charge[nelectron]/F",bufferSize); 
            tree_->Branch("electron_energy",&electron_energy,"electron_energy[nelectron]/F",bufferSize);
            tree_->Branch("electron_EtFromCaloEn",&electron_EtFromCaloEn,"electron_EtFromCaloEn[nelectron]/F",bufferSize);
            tree_->Branch("electron_isEB",&electron_isEB,"electron_isEB[nelectron]/F", bufferSize); 
            tree_->Branch("electron_isEE",&electron_isEE,"electron_isEE[nelectron]/F",bufferSize); 
            tree_->Branch("electron_ecalEnergy",&electron_ecalEnergy,"electron_ecalEnergy[nelectron]/F",bufferSize); 
            tree_->Branch("electron_isPassConversionVeto", &electron_isPassConversionVeto, "electron_isPassConversionVeto[nelectron]/F" ,bufferSize);
            tree_->Branch("electron_convDist",&electron_convDist,"electron_convDist[nelectron]" ,bufferSize); 
            tree_->Branch("electron_convFlags",&electron_convFlags,"electron_convFlags[nelectron]/I",bufferSize); 
            tree_->Branch("electron_convRadius",&electron_convRadius,"electron_convRadius[nelectron]/F",bufferSize); 
            tree_->Branch("electron_hadronicOverEm",&electron_hadronicOverEm,"electron_hadronicOverEm[nelectron]/F",bufferSize);
            tree_->Branch("electron_ecalDrivenSeed",&electron_ecalDrivenSeed,"electron_ecalDrivenSeed[nelectron]/F",bufferSize);

    // superCluster block

            tree_->Branch("electron_SC_energy",&electron_SC_energy,"electron_SC_energy[nelectron]/F",bufferSize); 
            tree_->Branch("electron_SC_deta",&electron_SC_deta,"electron_SC_deta[nelectron]/F",bufferSize); 
            tree_->Branch("electron_SC_dphi",&electron_SC_dphi,"electron_SC_dphi[nelectron]/F",bufferSize);
            tree_->Branch("electron_SC_et",&electron_SC_et,"electron_SC_et[nelectron]/F",bufferSize);
            tree_->Branch("electron_SC_eSuperClusterOverP",&electron_SC_eSuperClusterOverP,"electron_SC_eSuperClusterOverP[nelectron]/F",bufferSize); 
            tree_->Branch("electron_scE1x5Overe5x5",&electron_scE1x5Overe5x5,"electron_scE1x5Overe5x5[nelectron]/F",bufferSize); 
            tree_->Branch("electron_scE2x5MaxOvere5x5",&electron_scE2x5MaxOvere5x5,"electron_scE2x5MaxOvere5x5[nelectron]/F",bufferSize); 
            tree_->Branch("electron_scE5x5",&electron_scE5x5,"electron_scE5x5[nelectron]/F",bufferSize); 
            tree_->Branch("electron_scE5x5Rel",&electron_scE5x5Rel,"electron_scE5x5Rel[nelectron]/F",bufferSize); 
            tree_->Branch("electron_scPixCharge",&electron_scPixCharge,"electron_scPixCharge[nelectron]/F",bufferSize); 
            tree_->Branch("electron_scSigmaEtaEta",&electron_scSigmaEtaEta,"electron_scSigmaEtaEta[nelectron]/F",bufferSize);
            tree_->Branch("electron_scSigmaIEtaIEta",&electron_scSigmaIEtaIEta,"electron_scSigmaIEtaIEta[nelectron]/F",bufferSize);  
            tree_->Branch("electron_superClusterFbrem",&electron_superClusterFbrem,"electron_superClusterFbrem/F",bufferSize); 

            tree_->Branch("electron_2dIP",&electron_2dIP,"electron_2dIP[nelectron]/F",bufferSize); 
            tree_->Branch("electron_2dIPSig",&electron_2dIPSig,"electron_2dIPSig[nelectron]/F",bufferSize);
            tree_->Branch("electron_3dIP",&electron_3dIP,"electron_3dIP[nelectron]/F",bufferSize); 
            tree_->Branch("electron_3dIPSig",&electron_3dIPSig,"electron_3dIPSig[nelectron]/F",bufferSize); 
            tree_->Branch("electron_eSeedClusterOverP",&electron_eSeedClusterOverP,"electron_eSeedClusterOverP[nelectron]/F",bufferSize);
            tree_->Branch("electron_eSeedClusterOverPout",&electron_eSeedClusterOverPout,"electron_eSeedClusterOverPout[nelectron]/F",bufferSize);
            tree_->Branch("electron_eSuperClusterOverP",&electron_eSuperClusterOverP,"electron_eSuperClusterOverP[nelectron]/F",bufferSize);
            tree_->Branch("electron_eTopOvere5x5",&electron_eTopOvere5x5,"electron_eTopOvere5x5[nelectron]/F",bufferSize); 

            tree_->Branch("electron_deltaEtaEleClusterTrackAtCalo",&electron_deltaEtaEleClusterTrackAtCalo,"electron_deltaEtaEleClusterTrackAtCalo[nelectron]/F",bufferSize); 
            tree_->Branch("electron_deltaEtaSeedClusterTrackAtCalo",&electron_deltaEtaSeedClusterTrackAtCalo,"electron_deltaEtaSeedClusterTrackAtCalo[nelectron]/F",bufferSize);
            tree_->Branch("electron_deltaPhiSeedClusterTrackAtCalo",&electron_deltaPhiSeedClusterTrackAtCalo,"electron_deltaPhiSeedClusterTrackAtCalo[nelectron]/F",bufferSize); 
            tree_->Branch("electron_deltaEtaSeedClusterTrackAtVtx",&electron_deltaPhiSeedClusterTrackAtCalo,"electron_deltaPhiSeedClusterTrackAtCalo[nelectron]/F",bufferSize); 
            tree_->Branch("electron_deltaEtaSuperClusterTrackAtVtx",&electron_deltaEtaSuperClusterTrackAtVtx,"electron_deltaEtaSuperClusterTrackAtVtx[nelectron]/F",bufferSize);
            tree_->Branch("electron_deltaPhiEleClusterTrackAtCalo",&electron_deltaPhiEleClusterTrackAtCalo,"electron_deltaPhiEleClusterTrackAtCalo[nelectron]/F",bufferSize); 
            tree_->Branch("electron_deltaPhiSuperClusterTrackAtVtx",&electron_deltaPhiSuperClusterTrackAtVtx,"electron_deltaPhiSuperClusterTrackAtVtx[nelectron]/F",bufferSize);
            tree_->Branch("electron_sCseedEta",&electron_sCseedEta,"electron_sCseedEta[nelectron]/F",bufferSize);  

    // electron gsf variables. 
            tree_->Branch("electron_EtaRel",&electron_EtaRel,"electron_EtaRel[nelectron]/F",bufferSize); 
            tree_->Branch("electron_dxy",&electron_dxy,"electron_dxy[nelectron]/F",bufferSize); 
            tree_->Branch("electron_dz",&electron_dz,"electron_dz[nelectron]/F",bufferSize);
            tree_->Branch("electron_nbOfMissingHits",&electron_nbOfMissingHits,"electron_nbOfMissingHits[nelectron]/F",bufferSize); 
            tree_->Branch("electron_gsfCharge",&electron_gsfCharge,"electron_gsfCharge[nelectron]/F",bufferSize);


            tree_->Branch("electron_numberOfBrems",&electron_numberOfBrems,"electron_numberOfBrems[nelectron]/I",bufferSize);
            tree_->Branch("electron_trackFbrem",&electron_trackFbrem,"electron_trackFbrem[nelectron]/F",bufferSize); 
            tree_->Branch("electron_fbrem",&electron_fbrem,"electron_fbrem[nelectron]/F",bufferSize); 
            tree_->Branch("electron_e5x5",&electron_e5x5,"electron_e5x5[nelectron]/F",bufferSize); 
            tree_->Branch("electron_e5x5Rel",&electron_e5x5Rel,"electron_e5x5Rel[nelectron]/F",bufferSize); 
            tree_->Branch("electron_e1x5Overe5x5",&electron_e1x5Overe5x5,"electron_e1x5Overe5x5[nelectron]/F",bufferSize); 
            tree_->Branch("electron_e2x5MaxOvere5x5",&electron_e2x5MaxOvere5x5,"electron_e2x5MaxOvere5x5[nelectron]/F",bufferSize);

    // 5*5 cells cluster 
    //

            tree_->Branch("electron_full5x5_e5x5",&electron_full5x5_e5x5,"electron_full5x5_e5x5[nelectron]/F",bufferSize);
            tree_->Branch("electron_full5x5_e5x5Rel",&electron_full5x5_e5x5Rel,"electron_full5x5_e5x5Rel[nelectron]/F",bufferSize); 
            tree_->Branch("electron_full5x5_sigmaIetaIeta",&electron_full5x5_sigmaIetaIeta,"electron_full5x5_sigmaIetaIeta[nelectron]/F",bufferSize);
            tree_->Branch("electron_full5x5_e1x5Overe5x5",&electron_full5x5_e1x5Overe5x5,"electron_full5x5_e1x5Overe5x5[nelectron]/F",bufferSize);
            tree_->Branch("electron_full5x5_e2x5BottomOvere5x5",&electron_full5x5_e2x5BottomOvere5x5,"electron_full5x5_e2x5BottomOvere5x5[nelectron]/F",bufferSize);
            tree_->Branch("electron_full5x5_e2x5LeftOvere5x5",&electron_full5x5_e2x5LeftOvere5x5,"electron_full5x5_e2x5LeftOvere5x5[nelectron]/F",bufferSize);
            tree_->Branch("electron_full5x5_e2x5MaxOvere5x5",&electron_full5x5_e2x5MaxOvere5x5,"electron_full5x5_e2x5MaxOvere5x5[nelectron]/F",bufferSize);
            tree_->Branch("electron_full5x5_e2x5RightOvere5x5",&electron_full5x5_e2x5RightOvere5x5,"electron_full5x5_e2x5RightOvere5x5[nelectron]/F",bufferSize);
            tree_->Branch("electron_full5x5_e2x5TopOvere5x5",&electron_full5x5_e2x5TopOvere5x5,"electron_full5x5_e2x5TopOvere5x5[nelectron]/F",bufferSize);



            tree_->Branch("electron_full5x5_eBottomOvere5x5",&electron_full5x5_eBottomOvere5x5,"electron_full5x5_eBottomOvere5x5[nelectron]/F",bufferSize);
            tree_->Branch("electron_full5x5_eLeftOvere5x5",&electron_full5x5_eLeftOvere5x5,"electron_full5x5_eLeftOvere5x5[nelectron]/F",bufferSize);
            tree_->Branch("electron_full5x5_eRightOvere5x5",&electron_full5x5_eRightOvere5x5,"electron_full5x5_eRightOvere5x5[nelectron]/F",bufferSize);
            tree_->Branch("electron_full5x5_eTopOvere5x5",&electron_full5x5_eTopOvere5x5,"electron_full5x5_eTopOvere5x5[nelectron]/F",bufferSize);
            tree_->Branch("electron_full5x5_hcalDepth1OverEcal",&electron_full5x5_hcalDepth1OverEcal,"electron_full5x5_hcalDepth1OverEcal[nelectron]/F",bufferSize);
            tree_->Branch("electron_full5x5_hcalDepth1OverEcalBc",&electron_full5x5_hcalDepth1OverEcalBc,"electron_full5x5_hcalDepth1OverEcalBc[nelectron]/F",bufferSize);
            tree_->Branch("electron_full5x5_hcalDepth2OverEcal",&electron_full5x5_hcalDepth2OverEcal,"electron_full5x5_hcalDepth2OverEcal[nelectron]/F",bufferSize);
            tree_->Branch("electron_full5x5_hcalDepth2OverEcalBc",&electron_full5x5_hcalDepth2OverEcalBc,"electron_full5x5_hcalDepth2OverEcalBc[nelectron]/F",bufferSize);
            tree_->Branch("electron_full5x5_hcalOverEcal",&electron_full5x5_hcalOverEcal,"electron_full5x5_hcalOverEcal[nelectron]/F",bufferSize);
            tree_->Branch("electron_full5x5_hcalOverEcalBc",&electron_full5x5_hcalOverEcalBc, "electron_full5x5_hcalOverEcalBc[nelectron]/F",bufferSize);   
            tree_->Branch("electron_full5x5_r9",&electron_full5x5_r9,"electron_full5x5_r9[nelectron]/F",bufferSize);



    // Isolation block
            tree_->Branch("electron_neutralHadronIso",&electron_neutralHadronIso,"electron_neutralHadronIso[nelectron]/F",bufferSize); 
            tree_->Branch("electron_particleIso",&electron_particleIso,"electron_particleIso[nelectron]/F",bufferSize); 
            tree_->Branch("electron_photonIso",&electron_photonIso,"electron_photonIso[nelectron]/F",bufferSize);
            tree_->Branch("electron_puChargedHadronIso",&electron_puChargedHadronIso,"electron_puChargedHadronIso[nelectron]/F",bufferSize); 
            tree_->Branch("electron_trackIso",&electron_trackIso,"electron_trackIso[nelectron]/F",bufferSize);  
            tree_->Branch("electron_hcalDepth1OverEcal",&electron_hcalDepth1OverEcal,"electron_hcalDepth1OverEcal[nelectron]/F",bufferSize); 
            tree_->Branch("electron_hcalDepth2OverEcal",&electron_hcalDepth2OverEcal,"electron_hcalDepth2OverEcal[nelectron]/F",bufferSize); 
            tree_->Branch("electron_ecalPFClusterIso",&electron_ecalPFClusterIso,"electron_ecalPFClusterIso[nelectron]/F",bufferSize);
            tree_->Branch("electron_hcalPFClusterIso",&electron_hcalPFClusterIso,"electron_hcalPFClusterIso[nelectron]/F",bufferSize);  
            tree_->Branch("electron_dr03TkSumPt",&electron_dr03TkSumPt,"electron_dr03TkSumPt[nelectron]/F",bufferSize); 

            tree_->Branch("electron_dr03EcalRecHitSumEt",&electron_dr03EcalRecHitSumEt,"electron_dr03EcalRecHitSumEt[nelectron]/F",bufferSize); 
            tree_->Branch("electron_dr03HcalDepth1TowerSumEt",&electron_dr03HcalDepth1TowerSumEt,"electron_dr03HcalDepth1TowerSumEt[nelectron]/F",bufferSize);  
            tree_->Branch("electron_dr03HcalDepth1TowerSumEtBc",&electron_dr03HcalDepth1TowerSumEtBc,"electron_dr03HcalDepth1TowerSumEtBc[nelectron]/F",bufferSize); 
            tree_->Branch("electron_dr03HcalDepth2TowerSumEt",&electron_dr03HcalDepth2TowerSumEt,"electron_dr03HcalDepth2TowerSumEt[nelectron]/F",bufferSize); 
            tree_->Branch("electron_dr03HcalDepth2TowerSumEtBc",&electron_dr03HcalDepth2TowerSumEtBc,"electron_dr03HcalDepth2TowerSumEtBc[nelectron]/F",bufferSize); 
            tree_->Branch("electron_pfSumPhotonEt",&electron_pfSumPhotonEt,"electron_pfSumPhotonEt[nelectron]/F",bufferSize); 
            tree_->Branch("electron_pfSumChargedHadronPt",&electron_pfSumChargedHadronPt,"electron_pfSumChargedHadronPt[nelectron]/F",bufferSize); 
            tree_->Branch("electron_pfSumNeutralHadronEt",&electron_pfSumNeutralHadronEt,"electron_pfSumNeutralHadronEt[nelectron]/F",bufferSize); 
            tree_->Branch("electron_pfSumPUPt",&electron_pfSumPUPt,"electron_pfSumPUPt[nelectron]/F",bufferSize);

            tree_->Branch("electron_dr04EcalRecHitSumEt",&electron_dr04EcalRecHitSumEt,"electron_dr04EcalRecHitSumEt[nelectron]/F",bufferSize);  
            tree_->Branch("electron_dr04HcalDepth1TowerSumEt",&electron_dr04HcalDepth1TowerSumEt,"electron_dr04HcalDepth1TowerSumEt[nelectron]",bufferSize);  
            tree_->Branch("electron_dr04HcalDepth1TowerSumEtBc",&electron_dr04HcalDepth1TowerSumEtBc,"electron_dr04HcalDepth1TowerSumEtBc[nelectron]/F",bufferSize);
            tree_->Branch("electron_dr04HcalDepth2TowerSumEt",&electron_dr04HcalDepth2TowerSumEt,"electron_dr04HcalDepth2TowerSumEt[nelectron]/F",bufferSize); 
            tree_->Branch("electron_dr04HcalDepth2TowerSumEtBc",&electron_dr04HcalDepth2TowerSumEtBc,"electron_dr04HcalDepth2TowerSumEtBc[nelectron]/F",bufferSize);
            tree_->Branch("electron_dr04HcalTowerSumEt",&electron_dr04HcalTowerSumEt,"electron_dr04HcalTowerSumEt[nelectron]/F",bufferSize);
            tree_->Branch("electron_dr04HcalTowerSumEtBc",&electron_dr04HcalTowerSumEtBc,"electron_dr04HcalTowerSumEtBc[nelectron]/F",bufferSize);
		    
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
        
        int ientry_;
        
        static constexpr int maxEntries = 250; //25*10 -> allows for a maximum of 10 jets per event
        
        unsigned int nJet;
        float Jet_eta[maxEntries];
        float Jet_pt[maxEntries];
        unsigned int Jet_jetId[maxEntries];
        unsigned int Jet_nConstituents[maxEntries];
        unsigned int Jet_cleanmask[maxEntries];
        float Jet_forDA[maxEntries];
        int Jet_genJetIdx[maxEntries];

        float GenJet_pt[maxEntries];
        
        unsigned int njetorigin;
        int jetorigin_isPU[maxEntries];
        int jetorigin_isB[maxEntries];
        int jetorigin_isBB[maxEntries];
        int jetorigin_isGBB[maxEntries];
        int jetorigin_isLeptonic_B[maxEntries];
        int jetorigin_isLeptonic_C[maxEntries];
        int jetorigin_isC[maxEntries];
        int jetorigin_isCC[maxEntries];
        int jetorigin_isGCC[maxEntries];
        int jetorigin_isS[maxEntries];
        int jetorigin_isUD[maxEntries];
        int jetorigin_isG[maxEntries];

        int jetorigin_isLLP_RAD[maxEntries]; 
        int jetorigin_isLLP_MU[maxEntries]; 
        int jetorigin_isLLP_E[maxEntries]; 
        int jetorigin_isLLP_Q[maxEntries];
        int jetorigin_isLLP_QMU[maxEntries]; 
        int jetorigin_isLLP_QE[maxEntries]; 
        int jetorigin_isLLP_QQ[maxEntries]; 
        int jetorigin_isLLP_QQMU[maxEntries]; 
        int jetorigin_isLLP_QQE[maxEntries]; 
        int jetorigin_isLLP_B[maxEntries]; 
        int jetorigin_isLLP_BMU[maxEntries];
        int jetorigin_isLLP_BE[maxEntries]; 
        int jetorigin_isLLP_BB[maxEntries]; 
        int jetorigin_isLLP_BBMU[maxEntries]; 
        int jetorigin_isLLP_BBE[maxEntries]; 
        int jetorigin_isUndefined[maxEntries];
        
        float jetorigin_displacement[maxEntries];
        float jetorigin_decay_angle[maxEntries];
        float jetorigin_displacement_xy[maxEntries];
        float jetorigin_displacement_z[maxEntries]; 
        float jetorigin_betagamma[maxEntries];
        int   jetorigin_partonFlavor[maxEntries];
        int   jetorigin_hadronFlavor[maxEntries];
        int   jetorigin_llpId[maxEntries];
        float jetorigin_llp_mass[maxEntries];
        float jetorigin_llp_pt[maxEntries];
        

        unsigned int nglobal;
        float global_pt[maxEntries];
        float global_eta[maxEntries];
        float global_mass[maxEntries];
        int global_n60[maxEntries];
        int global_n90[maxEntries];
        float global_chargedEmEnergyFraction[maxEntries];
        float global_chargedHadronEnergyFraction[maxEntries];
        float global_chargedMuEnergyFraction[maxEntries];
        float global_electronEnergyFraction[maxEntries];

        float global_tau1[maxEntries];
        float global_tau2[maxEntries];
        float global_tau3[maxEntries];
    
        float global_relMassDropMassAK[maxEntries];
        float global_relMassDropMassCA[maxEntries];
        float global_relSoftDropMassAK[maxEntries];
        float global_relSoftDropMassCA[maxEntries];
    
        float global_thrust[maxEntries];
        float global_sphericity[maxEntries];
        float global_circularity[maxEntries];
        float global_isotropy[maxEntries];
        float global_eventShapeC[maxEntries];
        float global_eventShapeD[maxEntries];

        float xsecweight;
        float processId;
        float isData;
        
        unsigned int nlength;
        int length_cpf[maxEntries];
        int length_npf[maxEntries];
        int length_sv[maxEntries];
        int length_muon[maxEntries];
        int length_electron[maxEntries];
        
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
        float cpf_deta[maxEntries];
        float cpf_dphi[maxEntries];
        float cpf_drminsv[maxEntries];
        float cpf_vertex_association[maxEntries];
        float cpf_fromPV[maxEntries];
        float cpf_puppi_weight[maxEntries];
        float cpf_track_chi2[maxEntries];
        float cpf_track_ndof[maxEntries];
        float cpf_track_quality[maxEntries];
        float cpf_relmassdrop[maxEntries];
        
        int cpf_matchedSV[maxEntries];
        int cpf_matchedMuon[maxEntries];
        int cpf_matchedElectron[maxEntries];
	
    	int cpf_track_numberOfValidPixelHits[maxEntries];
    	int cpf_track_pixelLayersWithMeasurement[maxEntries];
    	int cpf_track_numberOfValidStripHits[maxEntries];
    	int cpf_track_stripLayersWithMeasurement[maxEntries]; 
        
        
        unsigned int ncsv;
        float csv_trackSumJetEtRatio[maxEntries];
        float csv_trackSumJetDeltaR[maxEntries];
        float csv_vertexCategory[maxEntries];
        float csv_trackSip2dValAboveCharm[maxEntries];
        float csv_trackSip2dSigAboveCharm[maxEntries];
        float csv_trackSip3dValAboveCharm[maxEntries];
        float csv_trackSip3dSigAboveCharm[maxEntries];
        float csv_jetNSelectedTracks[maxEntries];
        float csv_jetNTracksEtaRel[maxEntries];
        

        unsigned int nnpf;
        float npf_ptrel[maxEntries];
        float npf_deta[maxEntries];
        float npf_dphi[maxEntries];
        float npf_deltaR[maxEntries];
        float npf_isGamma[maxEntries];
        float npf_hcal_fraction[maxEntries];
        float npf_drminsv[maxEntries];
        float npf_puppi_weight[maxEntries];
        float npf_relmassdrop[maxEntries];
        
        unsigned int nsv;
        float sv_ptrel[maxEntries];
        float sv_deta[maxEntries];
        float sv_dphi[maxEntries];
        float sv_mass[maxEntries];
        float sv_deltaR[maxEntries];
        float sv_ntracks[maxEntries];
        float sv_chi2[maxEntries];
        float sv_ndof[maxEntries];
        float sv_dxy[maxEntries];
        float sv_dxysig[maxEntries];
        float sv_d3d[maxEntries];
        float sv_d3dsig[maxEntries];
        float sv_costhetasvpv[maxEntries];
        float sv_enratio[maxEntries];

        unsigned int nmuon;
        float muon_isGlobal[maxEntries] ; 
        float muon_isTight[maxEntries] ; 
        float muon_isMedium[maxEntries] ; 
        float muon_isLoose[maxEntries] ; 
        float muon_isStandAlone[maxEntries] ;

        float muon_ptrel [maxEntries];
        float muon_EtaRel[maxEntries];
        float muon_deta[maxEntries];
        float muon_dphi[maxEntries];
        float muon_charge [maxEntries]; 
        float muon_energy[maxEntries];
        float muon_jetDeltaR [maxEntries]; 
        float muon_numberOfMatchedStations [maxEntries];

        float muon_2dIp [maxEntries]; 
        float muon_2dIpSig [maxEntries];
        float muon_3dIp [maxEntries]; 
        float muon_3dIpSig [maxEntries]; 

        float muon_dxy [maxEntries]; 
        float muon_dxyError [maxEntries]; 
        float muon_dxySig [maxEntries]; 
        float muon_dz [maxEntries]; 
        float muon_dzError [maxEntries]; 
        float muon_numberOfValidPixelHits[maxEntries]; 
        float muon_numberOfpixelLayersWithMeasurement[maxEntries]; 

        float muon_chi2 [maxEntries]; 
        float muon_ndof [maxEntries]; 

        float muon_caloIso [maxEntries]; 
        float muon_ecalIso [maxEntries]; 
        float muon_hcalIso [maxEntries];

        float muon_sumPfChHadronPt [maxEntries]; 
        float muon_sumPfNeuHadronEt [maxEntries]; 
        float muon_Pfpileup [maxEntries]; 
        float muon_sumPfPhotonEt [maxEntries]; 

        float muon_sumPfChHadronPt03 [maxEntries]; 
        float muon_sumPfNeuHadronEt03 [maxEntries]; 
        float muon_Pfpileup03 [maxEntries]; 
        float muon_sumPfPhotonEt03 [maxEntries]; 

        float muon_sumChHadronPt [maxEntries]; 
        float muon_sumNeuHadronEt [maxEntries]; 
        float muon_pileup [maxEntries]; 
        float muon_sumPhotonEt [maxEntries]; 

        float muon_timeAtIpInOut [maxEntries]; 
        float muon_timeAtIpInOutErr [maxEntries]; 
        float muon_timeAtIpOutIn [maxEntries]; 

        unsigned int nelectron;

        unsigned int nelec;
	int   electron_jetIdx[maxEntries];
        float electron_ptrel[maxEntries];
        float electron_jetDeltaR[maxEntries]; 
        float electron_deta[maxEntries];
        float electron_dphi[maxEntries];
        float electron_charge[maxEntries]; 
        float electron_energy[maxEntries];
        float electron_EtFromCaloEn[maxEntries];
        float electron_isEB[maxEntries]; 
        float electron_isEE[maxEntries]; 
        float electron_ecalEnergy[maxEntries]; 
        float electron_isPassConversionVeto[maxEntries];
        float electron_convDist[maxEntries]; 
        int   electron_convFlags[maxEntries]; 
        float electron_convRadius[maxEntries]; 
        float electron_hadronicOverEm[maxEntries];
        float electron_ecalDrivenSeed[maxEntries];

    // superCluster block

        float electron_SC_energy[maxEntries]; 
        float electron_SC_deta[maxEntries]; 
        float electron_SC_dphi[maxEntries];
        float electron_SC_et[maxEntries];
        float electron_SC_eSuperClusterOverP[maxEntries]; 
        float electron_scE1x5Overe5x5[maxEntries]; 
        float electron_scE2x5MaxOvere5x5[maxEntries]; 
        float electron_scE5x5[maxEntries]; 
        float electron_scE5x5Rel[maxEntries]; 
        float electron_scPixCharge[maxEntries]; 
        float electron_scSigmaEtaEta[maxEntries];
        float electron_scSigmaIEtaIEta[maxEntries];  
        float electron_superClusterFbrem[maxEntries]; 

        float electron_2dIP[maxEntries]; 
        float electron_2dIPSig[maxEntries];
        float electron_3dIP[maxEntries]; 
        float electron_3dIPSig[maxEntries]; 
        float electron_eSeedClusterOverP[maxEntries];
        float electron_eSeedClusterOverPout[maxEntries];
        float electron_eSuperClusterOverP[maxEntries];
        float electron_eTopOvere5x5[maxEntries]; 

        float electron_deltaEtaEleClusterTrackAtCalo[maxEntries]; 
        float electron_deltaEtaSeedClusterTrackAtCalo[maxEntries];
        float electron_deltaPhiSeedClusterTrackAtCalo[maxEntries]; 
        float electron_deltaEtaSeedClusterTrackAtVtx[maxEntries]; 
        float electron_deltaEtaSuperClusterTrackAtVtx[maxEntries];
        float electron_deltaPhiEleClusterTrackAtCalo[maxEntries]; 
        float electron_deltaPhiSuperClusterTrackAtVtx[maxEntries];
        float electron_sCseedEta[maxEntries];  

    // electron gsf variables. 
        float electron_EtaRel[maxEntries]; 
        float electron_dxy[maxEntries]; 
        float electron_dz[maxEntries];
        float electron_nbOfMissingHits[maxEntries]; 
        float electron_gsfCharge[maxEntries];


        int   electron_numberOfBrems[maxEntries];
        float electron_trackFbrem[maxEntries]; 
        float electron_fbrem[maxEntries]; 
        float electron_e5x5[maxEntries]; 
        float electron_e5x5Rel[maxEntries]; 
        float electron_e1x5Overe5x5[maxEntries]; 
        float electron_e2x5MaxOvere5x5[maxEntries];

    // 5*5 cells cluster 
    //

        float electron_full5x5_e5x5[maxEntries];
        float electron_full5x5_e5x5Rel[maxEntries]; 
        float electron_full5x5_sigmaIetaIeta[maxEntries];
        float electron_full5x5_e1x5Overe5x5[maxEntries];
        float electron_full5x5_e2x5BottomOvere5x5[maxEntries];
        float electron_full5x5_e2x5LeftOvere5x5[maxEntries];
        float electron_full5x5_e2x5MaxOvere5x5[maxEntries];
        float electron_full5x5_e2x5RightOvere5x5[maxEntries];
        float electron_full5x5_e2x5TopOvere5x5[maxEntries];



        float electron_full5x5_eBottomOvere5x5[maxEntries];
        float electron_full5x5_eLeftOvere5x5[maxEntries];
        float electron_full5x5_eRightOvere5x5[maxEntries];
        float electron_full5x5_eTopOvere5x5[maxEntries];
        float electron_full5x5_hcalDepth1OverEcal[maxEntries];
        float electron_full5x5_hcalDepth1OverEcalBc[maxEntries];
        float electron_full5x5_hcalDepth2OverEcal[maxEntries];
        float electron_full5x5_hcalDepth2OverEcalBc[maxEntries];
        float electron_full5x5_hcalOverEcal[maxEntries];
        float electron_full5x5_hcalOverEcalBc[maxEntries];   
        float electron_full5x5_r9[maxEntries];



    // Isolation block
        float electron_neutralHadronIso[maxEntries]; 
        float electron_particleIso [maxEntries]; 
        float electron_photonIso[maxEntries];
        float electron_puChargedHadronIso[maxEntries]; 
        float electron_trackIso[maxEntries];  
        float electron_hcalDepth1OverEcal[maxEntries]; 
        float electron_hcalDepth2OverEcal[maxEntries]; 
        float electron_ecalPFClusterIso[maxEntries];
        float electron_hcalPFClusterIso[maxEntries];  
        float electron_dr03TkSumPt[maxEntries]; 

        float electron_dr03EcalRecHitSumEt[maxEntries]; 
        float electron_dr03HcalDepth1TowerSumEt[maxEntries];  
        float electron_dr03HcalDepth1TowerSumEtBc[maxEntries]; 
        float electron_dr03HcalDepth2TowerSumEt[maxEntries]; 
        float electron_dr03HcalDepth2TowerSumEtBc[maxEntries]; 
        float electron_pfSumPhotonEt[maxEntries]; 
        float electron_pfSumChargedHadronPt[maxEntries]; 
        float electron_pfSumNeutralHadronEt[maxEntries]; 
        float electron_pfSumPUPt[maxEntries];

        float electron_dr04EcalRecHitSumEt[maxEntries];  
        float electron_dr04HcalDepth1TowerSumEt[maxEntries];  
        float electron_dr04HcalDepth1TowerSumEtBc[maxEntries];
        float electron_dr04HcalDepth2TowerSumEt[maxEntries]; 
        float electron_dr04HcalDepth2TowerSumEtBc[maxEntries];
        float electron_dr04HcalTowerSumEt[maxEntries];
        float electron_dr04HcalTowerSumEtBc[maxEntries];
	
        std::mt19937 randomGenerator_;
        std::uniform_real_distribution<> uniform_dist_;
        
        typedef exprtk::symbol_table<float> SymbolTable;
        typedef exprtk::expression<float> Expression;
        typedef exprtk::parser<float> Parser;
        

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

        float isLLP_RAD;
        float isLLP_MU;
        float isLLP_E;
        float isLLP_Q;
        float isLLP_QMU;
        float isLLP_QE;
        float isLLP_QQ;
        float isLLP_QQMU;
        float isLLP_QQE;
        float isLLP_B;
        float isLLP_BMU;
        float isLLP_BE;
        float isLLP_BB;
        float isLLP_BBMU;
        float isLLP_BBE;
        float isPU;
        float fromLLP;
        
        float rand;
        float pt;
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
            ientry_(-1),
            randomGenerator_(12345),
            uniform_dist_(0,1.)
        {
            tree_->SetBranchAddress("nJet",&nJet);
            tree_->SetBranchAddress("Jet_eta",&Jet_eta);
            tree_->SetBranchAddress("Jet_pt",&Jet_pt);
            tree_->SetBranchAddress("Jet_jetId",&Jet_jetId);
            tree_->SetBranchAddress("Jet_cleanmask",&Jet_cleanmask);
            tree_->SetBranchAddress("Jet_nConstituents",&Jet_nConstituents);
            tree_->SetBranchAddress("Jet_genJetIdx", &Jet_genJetIdx);
            tree_->SetBranchAddress("GenJet_pt", &GenJet_pt);
        
            if (addTruth)
            {
                tree_->SetBranchAddress("njetorigin",&njetorigin);
                
                tree_->SetBranchAddress("jetorigin_displacement",&jetorigin_displacement);
                tree_->SetBranchAddress("jetorigin_decay_angle",&jetorigin_decay_angle);
                tree_->SetBranchAddress("jetorigin_displacement_xy" , &jetorigin_displacement_xy); 
                tree_->SetBranchAddress("jetorigin_displacement_z" , &jetorigin_displacement_z); 
                tree_->SetBranchAddress("jetorigin_betagamma", &jetorigin_betagamma);
                
                tree_->SetBranchAddress("jetorigin_partonFlavor", &jetorigin_partonFlavor);
                tree_->SetBranchAddress("jetorigin_hadronFlavor", &jetorigin_hadronFlavor);
                tree_->SetBranchAddress("jetorigin_llpId", &jetorigin_llpId);
                tree_->SetBranchAddress("jetorigin_llp_mass", &jetorigin_llp_mass);
                tree_->SetBranchAddress("jetorigin_llp_pt", &jetorigin_llp_pt);
                
                tree_->SetBranchAddress("jetorigin_isUndefined",&jetorigin_isUndefined);
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
                tree_->SetBranchAddress("jetorigin_isPU",&jetorigin_isPU);

                tree_->SetBranchAddress("jetorigin_isLLP_RAD",&jetorigin_isLLP_RAD ) ; 
                tree_->SetBranchAddress("jetorigin_isLLP_MU",&jetorigin_isLLP_MU) ; 
                tree_->SetBranchAddress("jetorigin_isLLP_E",&jetorigin_isLLP_E ) ;
                tree_->SetBranchAddress("jetorigin_isLLP_Q",&jetorigin_isLLP_Q ) ; 
                tree_->SetBranchAddress("jetorigin_isLLP_QMU",&jetorigin_isLLP_QMU) ; 
                tree_->SetBranchAddress("jetorigin_isLLP_QE",&jetorigin_isLLP_QE) ; 
                tree_->SetBranchAddress("jetorigin_isLLP_QQ",&jetorigin_isLLP_QQ ); 
                tree_->SetBranchAddress("jetorigin_isLLP_QQMU",&jetorigin_isLLP_QQMU ); 
                tree_->SetBranchAddress("jetorigin_isLLP_QQE",&jetorigin_isLLP_QQE) ; 
                tree_->SetBranchAddress("jetorigin_isLLP_QQE",&jetorigin_isLLP_QQE ) ; 
                tree_->SetBranchAddress("jetorigin_isLLP_BMU",&jetorigin_isLLP_BMU) ; 
                tree_->SetBranchAddress("jetorigin_isLLP_BE",&jetorigin_isLLP_BE ) ; 
                tree_->SetBranchAddress("jetorigin_isLLP_BB",&jetorigin_isLLP_BB) ; 
                tree_->SetBranchAddress("jetorigin_isLLP_BBMU",&jetorigin_isLLP_BBMU) ; 
                tree_->SetBranchAddress("jetorigin_isLLP_BBE",&jetorigin_isLLP_BBE ) ; 

                
            }
            else
            {
                tree_->SetBranchAddress("Jet_forDA",&Jet_forDA);
                tree_->SetBranchAddress("xsecweight",&xsecweight);
                tree_->SetBranchAddress("isData",&isData);
                tree_->SetBranchAddress("processId",&processId);
            }
            
            tree_->SetBranchAddress("nlength",&nlength);
            tree_->SetBranchAddress("length_cpf",&length_cpf);
            tree_->SetBranchAddress("length_npf",&length_npf);
            tree_->SetBranchAddress("length_sv",&length_sv);
            tree_->SetBranchAddress("length_mu",&length_muon);
            tree_->SetBranchAddress("length_ele",&length_electron);

            tree_->SetBranchAddress("nglobal",&nglobal);
            tree_->SetBranchAddress("global_pt",&global_pt);
            tree_->SetBranchAddress("global_eta",&global_eta);
            
            tree_->SetBranchAddress("global_mass",&global_mass);
            tree_->SetBranchAddress("global_n60",&global_n60);
            tree_->SetBranchAddress("global_n90",&global_n90);
            tree_->SetBranchAddress("global_chargedEmEnergyFraction",&global_chargedEmEnergyFraction);
            tree_->SetBranchAddress("global_chargedHadronEnergyFraction",&global_chargedHadronEnergyFraction);
            tree_->SetBranchAddress("global_chargedMuEnergyFraction",&global_chargedMuEnergyFraction);
            tree_->SetBranchAddress("global_electronEnergyFraction",&global_electronEnergyFraction);

            tree_->SetBranchAddress("global_tau1",&global_tau1);
            tree_->SetBranchAddress("global_tau2",&global_tau2);
            tree_->SetBranchAddress("global_tau3",&global_tau3);
    
            tree_->SetBranchAddress("global_relMassDropMassAK",&global_relMassDropMassAK);
            tree_->SetBranchAddress("global_relMassDropMassCA",&global_relMassDropMassCA);
            tree_->SetBranchAddress("global_relSoftDropMassAK",&global_relSoftDropMassAK);
            tree_->SetBranchAddress("global_relSoftDropMassCA",&global_relSoftDropMassCA);
    
            tree_->SetBranchAddress("global_thrust",&global_thrust);
            tree_->SetBranchAddress("global_sphericity",&global_sphericity);
            tree_->SetBranchAddress("global_circularity",&global_circularity);
            tree_->SetBranchAddress("global_isotropy",&global_isotropy);
            tree_->SetBranchAddress("global_eventShapeC",&global_eventShapeC);
            tree_->SetBranchAddress("global_eventShapeD",&global_eventShapeD);
            
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
            tree_->SetBranchAddress("cpf_deta",&cpf_deta);
            tree_->SetBranchAddress("cpf_dphi",&cpf_dphi);
            tree_->SetBranchAddress("cpf_drminsv",&cpf_drminsv);
            tree_->SetBranchAddress("cpf_vertex_association",&cpf_vertex_association);
            tree_->SetBranchAddress("cpf_fromPV",&cpf_fromPV);
            tree_->SetBranchAddress("cpf_puppi_weight",&cpf_puppi_weight);
            tree_->SetBranchAddress("cpf_track_chi2",&cpf_track_chi2);
            tree_->SetBranchAddress("cpf_track_ndof",&cpf_track_ndof);
            tree_->SetBranchAddress("cpf_track_quality",&cpf_track_quality);
            tree_->SetBranchAddress("cpf_relmassdrop",&cpf_relmassdrop);
            
            tree_->SetBranchAddress("cpf_matchedSV",&cpf_matchedSV);
            tree_->SetBranchAddress("cpf_matchedMuon",&cpf_matchedMuon);
            tree_->SetBranchAddress("cpf_matchedElectron",&cpf_matchedElectron);
	    tree_->SetBranchAddress("cpf_track_numberOfValidPixelHits", &cpf_track_numberOfValidPixelHits);
	    tree_->SetBranchAddress("cpf_track_pixelLayersWithMeasurement", &cpf_track_pixelLayersWithMeasurement) ;
	    tree_->SetBranchAddress("cpf_track_numberOfValidStripHits" , &cpf_track_numberOfValidStripHits) ;
 	    tree_->SetBranchAddress("cpf_track_stripLayersWithMeasurement" , &cpf_track_stripLayersWithMeasurement);
            
          
            tree_->SetBranchAddress("nnpf",&nnpf);
            tree_->SetBranchAddress("npf_ptrel",&npf_ptrel);
            tree_->SetBranchAddress("npf_deta",&npf_deta);
            tree_->SetBranchAddress("npf_dphi",&npf_dphi);
            tree_->SetBranchAddress("npf_deltaR",&npf_deltaR);
            tree_->SetBranchAddress("npf_isGamma",&npf_isGamma);
            tree_->SetBranchAddress("npf_hcal_fraction",&npf_hcal_fraction);
            tree_->SetBranchAddress("npf_drminsv",&npf_drminsv);
            tree_->SetBranchAddress("npf_puppi_weight",&npf_puppi_weight);
            tree_->SetBranchAddress("npf_relmassdrop",&npf_relmassdrop);
            
            tree_->SetBranchAddress("nsv",&nsv);
            tree_->SetBranchAddress("sv_ptrel",&sv_ptrel);
            tree_->SetBranchAddress("sv_deta",&sv_deta);
            tree_->SetBranchAddress("sv_dphi",&sv_dphi);
            tree_->SetBranchAddress("sv_mass",&sv_mass);
            tree_->SetBranchAddress("sv_deltaR",&sv_deltaR);
            tree_->SetBranchAddress("sv_ntracks",&sv_ntracks);
            tree_->SetBranchAddress("sv_chi2",&sv_chi2);
            tree_->SetBranchAddress("sv_ndof",&sv_ndof);
            tree_->SetBranchAddress("sv_dxy",&sv_dxy);
            tree_->SetBranchAddress("sv_dxysig",&sv_dxysig);
            tree_->SetBranchAddress("sv_d3d",&sv_d3d);
            tree_->SetBranchAddress("sv_d3dsig",&sv_d3dsig);
            tree_->SetBranchAddress("sv_costhetasvpv",&sv_costhetasvpv);
            tree_->SetBranchAddress("sv_enratio",&sv_enratio);
		  
	    tree_->SetBranchAddress("nmuon",&nmuon); 
            tree_->SetBranchAddress("muon_isGlobal",&muon_isGlobal); 
            tree_->SetBranchAddress("muon_isTight",&muon_isTight); 
            tree_->SetBranchAddress("muon_isMedium",&muon_isMedium); 
            tree_->SetBranchAddress("muon_isLoose",&muon_isLoose); 
            tree_->SetBranchAddress("muon_isStandAlone",&muon_isStandAlone);

            tree_->SetBranchAddress("muon_ptrel", &muon_ptrel);
            tree_->SetBranchAddress("muon_EtaRel", &muon_EtaRel);
            tree_->SetBranchAddress("muon_dphi",&muon_dphi);
            tree_->SetBranchAddress("muon_deta",&muon_deta);
            tree_->SetBranchAddress("muon_charge",&muon_charge); 
            tree_->SetBranchAddress("muon_energy",&muon_energy);
            tree_->SetBranchAddress("muon_jetDeltaR",&muon_jetDeltaR); 
            tree_->SetBranchAddress("muon_numberOfMatchedStations",&muon_numberOfMatchedStations);

            tree_->SetBranchAddress("muon_2dIp", &muon_2dIp); 
            tree_->SetBranchAddress("muon_2dIpSig", &muon_2dIpSig);
            tree_->SetBranchAddress("muon_3dIp",&muon_3dIp); 
            tree_->SetBranchAddress("muon_3dIpSig",&muon_3dIpSig); 

            tree_->SetBranchAddress("muon_dxy", &muon_dxy); 
            tree_->SetBranchAddress("muon_dxyError", &muon_dxyError); 
            tree_->SetBranchAddress("muon_dxySig",&muon_dxySig); 
            tree_->SetBranchAddress("muon_dz",&muon_dz); 
            tree_->SetBranchAddress("muon_dzError", &muon_dzError); 
            tree_->SetBranchAddress("muon_numberOfValidPixelHits",&muon_numberOfValidPixelHits); 
            tree_->SetBranchAddress("muon_numberOfpixelLayersWithMeasurement", &muon_numberOfpixelLayersWithMeasurement); 

            tree_->SetBranchAddress("muon_chi2", &muon_chi2); 
            tree_->SetBranchAddress("muon_ndof",&muon_ndof); 

            tree_->SetBranchAddress("muon_caloIso",&muon_caloIso); 
            tree_->SetBranchAddress("muon_ecalIso",&muon_ecalIso); 
            tree_->SetBranchAddress("muon_hcalIso",&muon_hcalIso);

            tree_->SetBranchAddress("muon_sumPfChHadronPt",&muon_sumPfChHadronPt); 
            tree_->SetBranchAddress("muon_sumPfNeuHadronEt",&muon_sumPfNeuHadronEt); 
            tree_->SetBranchAddress("muon_Pfpileup",&muon_Pfpileup); 
            tree_->SetBranchAddress("muon_sumPfPhotonEt",&muon_sumPfPhotonEt); 

            tree_->SetBranchAddress("muon_sumPfChHadronPt03",&muon_sumPfChHadronPt03); 
            tree_->SetBranchAddress("muon_sumPfNeuHadronEt03",&muon_sumPfNeuHadronEt03); 
            tree_->SetBranchAddress("muon_Pfpileup03",&muon_Pfpileup03); 
            tree_->SetBranchAddress("muon_sumPfPhotonEt03",&muon_sumPfPhotonEt03); 
            tree_->SetBranchAddress("muon_sumChHadronPt",&muon_sumChHadronPt);
 
 
            tree_->SetBranchAddress("nelectron",&nelectron); 
 
//            tree_->SetBranchAddress("nelec",&nelec);
            tree_->SetBranchAddress("electron_jetIdx",&electron_jetIdx);
            tree_->SetBranchAddress("electron_ptrel",&electron_ptrel);
            tree_->SetBranchAddress("electron_jetDeltaR",&electron_jetDeltaR); 
            tree_->SetBranchAddress("electron_deta",&electron_deta);
            tree_->SetBranchAddress("electron_dphi",&electron_dphi);
            tree_->SetBranchAddress("electron_charge",&electron_charge); 
            tree_->SetBranchAddress("electron_energy",&electron_energy);
            tree_->SetBranchAddress("electron_EtFromCaloEn",&electron_EtFromCaloEn);
            tree_->SetBranchAddress("electron_isEB",&electron_isEB); 
            tree_->SetBranchAddress("electron_isEE",&electron_isEE); 
            tree_->SetBranchAddress("electron_ecalEnergy",&electron_ecalEnergy); 
            tree_->SetBranchAddress("electron_isPassConversionVeto", &electron_isPassConversionVeto);
            tree_->SetBranchAddress("electron_convDist",&electron_convDist); 
            tree_->SetBranchAddress("electron_convFlags",&electron_convFlags); 
            tree_->SetBranchAddress("electron_convRadius",&electron_convRadius); 
            tree_->SetBranchAddress("electron_hadronicOverEm",&electron_hadronicOverEm);
            tree_->SetBranchAddress("electron_ecalDrivenSeed",&electron_ecalDrivenSeed);

    // superCluster block

            tree_->SetBranchAddress("electron_SC_energy",&electron_SC_energy); 
            tree_->SetBranchAddress("electron_SC_deta",&electron_SC_deta); 
            tree_->SetBranchAddress("electron_SC_dphi",&electron_SC_dphi);
            tree_->SetBranchAddress("electron_SC_et",&electron_SC_et);
            tree_->SetBranchAddress("electron_SC_eSuperClusterOverP",&electron_SC_eSuperClusterOverP); 
            tree_->SetBranchAddress("electron_scE1x5Overe5x5",&electron_scE1x5Overe5x5); 
            tree_->SetBranchAddress("electron_scE2x5MaxOvere5x5",&electron_scE2x5MaxOvere5x5); 
            tree_->SetBranchAddress("electron_scE5x5",&electron_scE5x5); 
            tree_->SetBranchAddress("electron_scE5x5Rel",&electron_scE5x5Rel); 
            tree_->SetBranchAddress("electron_scPixCharge",&electron_scPixCharge); 
            tree_->SetBranchAddress("electron_scSigmaEtaEta",&electron_scSigmaEtaEta);
            tree_->SetBranchAddress("electron_scSigmaIEtaIEta",&electron_scSigmaIEtaIEta);  
            tree_->SetBranchAddress("electron_superClusterFbrem",&electron_superClusterFbrem); 

            tree_->SetBranchAddress("electron_2dIP",&electron_2dIP); 
            tree_->SetBranchAddress("electron_2dIPSig",&electron_2dIPSig);
            tree_->SetBranchAddress("electron_3dIP",&electron_3dIP); 
            tree_->SetBranchAddress("electron_3dIPSig",&electron_3dIPSig); 
            tree_->SetBranchAddress("electron_eSeedClusterOverP",&electron_eSeedClusterOverP);
            tree_->SetBranchAddress("electron_eSeedClusterOverPout",&electron_eSeedClusterOverPout);
            tree_->SetBranchAddress("electron_eSuperClusterOverP",&electron_eSuperClusterOverP);
            tree_->SetBranchAddress("electron_eTopOvere5x5",&electron_eTopOvere5x5); 

            tree_->SetBranchAddress("electron_deltaEtaEleClusterTrackAtCalo",&electron_deltaEtaEleClusterTrackAtCalo); 
            tree_->SetBranchAddress("electron_deltaEtaSeedClusterTrackAtCalo",&electron_deltaEtaSeedClusterTrackAtCalo);
            tree_->SetBranchAddress("electron_deltaPhiSeedClusterTrackAtCalo",&electron_deltaPhiSeedClusterTrackAtCalo); 
            tree_->SetBranchAddress("electron_deltaEtaSeedClusterTrackAtVtx",&electron_deltaPhiSeedClusterTrackAtCalo); 
            tree_->SetBranchAddress("electron_deltaEtaSuperClusterTrackAtVtx",&electron_deltaEtaSuperClusterTrackAtVtx);
            tree_->SetBranchAddress("electron_deltaPhiEleClusterTrackAtCalo",&electron_deltaPhiEleClusterTrackAtCalo); 
            tree_->SetBranchAddress("electron_deltaPhiSuperClusterTrackAtVtx",&electron_deltaPhiSuperClusterTrackAtVtx);
            tree_->SetBranchAddress("electron_sCseedEta",&electron_sCseedEta);  

    // electron gsf variables. 
            tree_->SetBranchAddress("electron_EtaRel",&electron_EtaRel); 
            tree_->SetBranchAddress("electron_dxy",&electron_dxy); 
            tree_->SetBranchAddress("electron_dz",&electron_dz);
            tree_->SetBranchAddress("electron_nbOfMissingHits",&electron_nbOfMissingHits); 
            tree_->SetBranchAddress("electron_gsfCharge",&electron_gsfCharge);


            tree_->SetBranchAddress("electron_numberOfBrems",&electron_numberOfBrems);
            tree_->SetBranchAddress("electron_trackFbrem",&electron_trackFbrem); 
            tree_->SetBranchAddress("electron_fbrem",&electron_fbrem); 
            tree_->SetBranchAddress("electron_e5x5",&electron_e5x5); 
            tree_->SetBranchAddress("electron_e5x5Rel",&electron_e5x5Rel); 
            tree_->SetBranchAddress("electron_e1x5Overe5x5",&electron_e1x5Overe5x5); 
            tree_->SetBranchAddress("electron_e2x5MaxOvere5x5",&electron_e2x5MaxOvere5x5);

    // 5*5 cells cluster 
    //

            tree_->SetBranchAddress("electron_full5x5_e5x5",&electron_full5x5_e5x5);
            tree_->SetBranchAddress("electron_full5x5_e5x5Rel",&electron_full5x5_e5x5Rel); 
            tree_->SetBranchAddress("electron_full5x5_sigmaIetaIeta",&electron_full5x5_sigmaIetaIeta);
            tree_->SetBranchAddress("electron_full5x5_e1x5Overe5x5",&electron_full5x5_e1x5Overe5x5);
            tree_->SetBranchAddress("electron_full5x5_e2x5BottomOvere5x5",&electron_full5x5_e2x5BottomOvere5x5);
            tree_->SetBranchAddress("electron_full5x5_e2x5LeftOvere5x5",&electron_full5x5_e2x5LeftOvere5x5);
            tree_->SetBranchAddress("electron_full5x5_e2x5MaxOvere5x5",&electron_full5x5_e2x5MaxOvere5x5);
            tree_->SetBranchAddress("electron_full5x5_e2x5RightOvere5x5",&electron_full5x5_e2x5RightOvere5x5);
            tree_->SetBranchAddress("electron_full5x5_e2x5TopOvere5x5",&electron_full5x5_e2x5TopOvere5x5);



            tree_->SetBranchAddress("electron_full5x5_eBottomOvere5x5",&electron_full5x5_eBottomOvere5x5);
            tree_->SetBranchAddress("electron_full5x5_eLeftOvere5x5",&electron_full5x5_eLeftOvere5x5);
            tree_->SetBranchAddress("electron_full5x5_eRightOvere5x5",&electron_full5x5_eRightOvere5x5);
            tree_->SetBranchAddress("electron_full5x5_eTopOvere5x5",&electron_full5x5_eTopOvere5x5);
            tree_->SetBranchAddress("electron_full5x5_hcalDepth1OverEcal",&electron_full5x5_hcalDepth1OverEcal);
            tree_->SetBranchAddress("electron_full5x5_hcalDepth1OverEcalBc",&electron_full5x5_hcalDepth1OverEcalBc);
            tree_->SetBranchAddress("electron_full5x5_hcalDepth2OverEcal",&electron_full5x5_hcalDepth2OverEcal);
            tree_->SetBranchAddress("electron_full5x5_hcalDepth2OverEcalBc",&electron_full5x5_hcalDepth2OverEcalBc);
            tree_->SetBranchAddress("electron_full5x5_hcalOverEcal",&electron_full5x5_hcalOverEcal);
            tree_->SetBranchAddress("electron_full5x5_hcalOverEcalBc",&electron_full5x5_hcalOverEcalBc);   
            tree_->SetBranchAddress("electron_full5x5_r9",&electron_full5x5_r9);



    // Isolation block
            tree_->SetBranchAddress("electron_neutralHadronIso",&electron_neutralHadronIso); 
            tree_->SetBranchAddress("electron_particleIso",&electron_particleIso); 
            tree_->SetBranchAddress("electron_photonIso",&electron_photonIso);
            tree_->SetBranchAddress("electron_puChargedHadronIso",&electron_puChargedHadronIso); 
            tree_->SetBranchAddress("electron_trackIso",&electron_trackIso);  
            tree_->SetBranchAddress("electron_hcalDepth1OverEcal",&electron_hcalDepth1OverEcal); 
            tree_->SetBranchAddress("electron_hcalDepth2OverEcal",&electron_hcalDepth2OverEcal); 
            tree_->SetBranchAddress("electron_ecalPFClusterIso",&electron_ecalPFClusterIso);
            tree_->SetBranchAddress("electron_hcalPFClusterIso",&electron_hcalPFClusterIso);  
            tree_->SetBranchAddress("electron_dr03TkSumPt",&electron_dr03TkSumPt); 

            tree_->SetBranchAddress("electron_dr03EcalRecHitSumEt",&electron_dr03EcalRecHitSumEt); 
            tree_->SetBranchAddress("electron_dr03HcalDepth1TowerSumEt",&electron_dr03HcalDepth1TowerSumEt);  
            tree_->SetBranchAddress("electron_dr03HcalDepth1TowerSumEtBc",&electron_dr03HcalDepth1TowerSumEtBc); 
            tree_->SetBranchAddress("electron_dr03HcalDepth2TowerSumEt",&electron_dr03HcalDepth2TowerSumEt); 
            tree_->SetBranchAddress("electron_dr03HcalDepth2TowerSumEtBc",&electron_dr03HcalDepth2TowerSumEtBc); 
            tree_->SetBranchAddress("electron_pfSumPhotonEt",&electron_pfSumPhotonEt); 
            tree_->SetBranchAddress("electron_pfSumChargedHadronPt",&electron_pfSumChargedHadronPt); 
            tree_->SetBranchAddress("electron_pfSumNeutralHadronEt",&electron_pfSumNeutralHadronEt); 
            tree_->SetBranchAddress("electron_pfSumPUPt",&electron_pfSumPUPt);

            tree_->SetBranchAddress("electron_dr04EcalRecHitSumEt",&electron_dr04EcalRecHitSumEt);  
            tree_->SetBranchAddress("electron_dr04HcalDepth1TowerSumEt",&electron_dr04HcalDepth1TowerSumEt);  
            tree_->SetBranchAddress("electron_dr04HcalDepth1TowerSumEtBc",&electron_dr04HcalDepth1TowerSumEtBc);
            tree_->SetBranchAddress("electron_dr04HcalDepth2TowerSumEt",&electron_dr04HcalDepth2TowerSumEt); 
            tree_->SetBranchAddress("electron_dr04HcalDepth2TowerSumEtBc",&electron_dr04HcalDepth2TowerSumEtBc);
            tree_->SetBranchAddress("electron_dr04HcalTowerSumEt",&electron_dr04HcalTowerSumEt);
            tree_->SetBranchAddress("electron_dr04HcalTowerSumEtBc",&electron_dr04HcalTowerSumEtBc);
 
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
            
            symbolTable_.add_variable("rand",rand);
            symbolTable_.add_variable("ctau",ctau);
	    
            symbolTable_.add_variable("pt",pt);
            symbolTable_.add_variable("isLLP_RAD" ,isLLP_RAD) ; 
            symbolTable_.add_variable("isLLP_MU" ,isLLP_MU) ; 
            symbolTable_.add_variable("isLLP_E",isLLP_E) ; 
            symbolTable_.add_variable("isLLP_Q" ,isLLP_Q) ; 
            symbolTable_.add_variable("isLLP_QMU" ,isLLP_QMU) ; 
            symbolTable_.add_variable("isLLP_QE" ,isLLP_QE) ; 
            symbolTable_.add_variable("isLLP_QQ" ,isLLP_QQ ) ; 
            symbolTable_.add_variable("isLLP_QQMU" ,isLLP_QQMU) ; 
            symbolTable_.add_variable("isLLP_QQE" ,isLLP_QQE) ; 
            symbolTable_.add_variable("isLLP_B" ,isLLP_B) ; 
            symbolTable_.add_variable("isLLP_BMU" ,isLLP_BMU) ; 
            symbolTable_.add_variable("isLLP_BE" ,isLLP_BE) ; 
            symbolTable_.add_variable("isLLP_BB" ,isLLP_BB) ; 
            symbolTable_.add_variable("isLLP_BBMU" ,isLLP_BBMU) ; 
            symbolTable_.add_variable("isLLP_BBE" ,isLLP_BBE) ; 
            symbolTable_.add_variable("isPU",isPU);

            
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
        
        inline int entry() const
        {
            return ientry_;
        }
        
        bool getEvent(int entry, bool force=false)
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
            return getEvent((ientry_+1)%entries());
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

            // Cut on pileup
            /*
            if (Jet_genJetIdx[jet] > -1)
            {
                if ((Jet_pt[jet] - GenJet_pt[Jet_genJetIdx[jet]])/GenJet_pt[Jet_genJetIdx[jet]] < -.75)
                {
                    //std::cout << Jet_pt[jet] << "; " << Jet_genJetIdx[jet] << "; " << jetorigin_fromLLP[jet] << std::endl;
                    return false;
                }
            }
            */
            
            if (this->njets()<jet)
            {
                std::cout<<"Not enough jets to unpack"<<std::endl;
                return false;
            }
            
            
            //do not apply jet ID; require at least 2 constituents & no overlap with leptons
            //garbage jets are anyway not considered since training is done on matched jets only
            if (Jet_nConstituents[jet]<2)
            {
                return false;
            }
            
            if (std::fabs(Jet_eta[jet])>2.4)
            {
                return false;
            }
            
            if (addTruth_)
            {
                if (jetorigin_isUndefined[jet]>0.5)
                {
                    return false;
                }

                isB = jetorigin_isB[jet];
                isBB = jetorigin_isBB[jet];
                isGBB = jetorigin_isGBB[jet];
                isLeptonic_B = jetorigin_isLeptonic_B[jet];
                isLeptonic_C = jetorigin_isLeptonic_C[jet];
                
                isC = jetorigin_isC[jet];
                isCC = jetorigin_isCC[jet];
                isGCC = jetorigin_isGCC[jet];
                
                isS = jetorigin_isS[jet];
                isUD = jetorigin_isUD[jet];
                isG = jetorigin_isG[jet];
 
                isLLP_RAD= jetorigin_isLLP_RAD[jet];  
                isLLP_MU= jetorigin_isLLP_MU[jet];  
                isLLP_E= jetorigin_isLLP_E[jet];  
                isLLP_Q= jetorigin_isLLP_Q[jet];  
                isLLP_QMU= jetorigin_isLLP_QMU[jet];  
                isLLP_QE= jetorigin_isLLP_QE[jet];  
                isLLP_QQ= jetorigin_isLLP_QQ[jet];  
                isLLP_QQMU= jetorigin_isLLP_QQMU[jet];  
                isLLP_QQE= jetorigin_isLLP_QQE[jet];  
                isLLP_B= jetorigin_isLLP_B[jet];  
                isLLP_BMU= jetorigin_isLLP_BMU[jet];  
                isLLP_BE= jetorigin_isLLP_BE[jet];  
                isLLP_BB= jetorigin_isLLP_BB[jet];  
                isLLP_BBMU= jetorigin_isLLP_BBMU[jet];  
                isLLP_BBE= jetorigin_isLLP_BBE[jet];  
                
                isPU = jetorigin_isPU[jet];
               
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
                isLLP_RAD = 0;
                isLLP_MU = 0;
                isLLP_E = 0;
                isLLP_Q = 0;
                isLLP_QMU = 0;
                isLLP_QE = 0;
                isLLP_QQ = 0;
                isLLP_QQMU = 0;
                isLLP_QQE = 0;
                isLLP_B = 0;
                isLLP_BMU = 0;
                isLLP_BE = 0;
                isLLP_BB = 0;
                isLLP_BBMU = 0;
                isLLP_BBE = 0;
                isPU = 0;
            }
            
            
            rand = uniform_dist_(randomGenerator_);
            //ctau = 1e9;
            pt = global_pt[jet];
            
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
            if  (jetorigin_isPU[jet]>0.5) return 12;
            if  (jetorigin_isLLP_RAD[jet]>0.5) return 13;
            if  (jetorigin_isLLP_MU[jet]>0.5) return 14;
            if  (jetorigin_isLLP_E[jet]>0.5) return 15;
            if  (jetorigin_isLLP_Q[jet]>0.5) return 16;
            if  (jetorigin_isLLP_QMU[jet]>0.5) return 17;
            if  (jetorigin_isLLP_QE[jet]>0.5) return 18;
            if  (jetorigin_isLLP_QQ[jet]>0.5) return 19;
            if  (jetorigin_isLLP_QQMU[jet]>0.5) return 20;
            if  (jetorigin_isLLP_QQE[jet]>0.5) return 21;
            if  (jetorigin_isLLP_B[jet]>0.5) return 22;
            if  (jetorigin_isLLP_BMU[jet]>0.5) return 23;
            if  (jetorigin_isLLP_BE[jet]>0.5) return 24;
            if  (jetorigin_isLLP_BB[jet]>0.5) return 25;
            if  (jetorigin_isLLP_BBMU[jet]>0.5) return 26;
            if  (jetorigin_isLLP_BBE[jet]>0.5) return 27;

            return -1;
        }
        
        bool unpackJet(
            unsigned int jet,
            UnpackedTree& unpackedTree
        )
        {
            if (this->njets()!=nglobal
                or this->njets()!=nlength 
                or this->njets()!=ncsv
            )
            {
                std::cout<<"Encountered weird event with unclear numbers of jets"<<std::endl;
                std::cout<<"\tnjets = "<<this->njets()<<std::endl;
                std::cout<<"\tnlength = "<<nlength<<std::endl;
                std::cout<<"\tnglobal = "<<nglobal<<std::endl;
                if (addTruth_)
                {
                    std::cout<<"\tnjetorigin = "<<njetorigin<<std::endl;
                }

                return false;
            }
            
            if (addTruth_)
            {
                unpackedTree.jetorigin_isUndefined = jetorigin_isUndefined[jet];
                unpackedTree.jetorigin_displacement = jetorigin_displacement[jet];
                unpackedTree.jetorigin_ctau = ctau;
                unpackedTree.jetorigin_decay_angle = jetorigin_decay_angle[jet];
                
                unpackedTree.jetorigin_isB = jetorigin_isB[jet];
                unpackedTree.jetorigin_isBB = jetorigin_isBB[jet];
                unpackedTree.jetorigin_isGBB = jetorigin_isGBB[jet];
                unpackedTree.jetorigin_isLeptonic_B = jetorigin_isLeptonic_B[jet];
                unpackedTree.jetorigin_isLeptonic_C = jetorigin_isLeptonic_C[jet];
                unpackedTree.jetorigin_isC = jetorigin_isC[jet];
                unpackedTree.jetorigin_isCC = jetorigin_isCC[jet];
                unpackedTree.jetorigin_isGCC = jetorigin_isGCC[jet];
                unpackedTree.jetorigin_isS = jetorigin_isS[jet];
                unpackedTree.jetorigin_isUD = jetorigin_isUD[jet];
                unpackedTree.jetorigin_isG = jetorigin_isG[jet];
                unpackedTree.jetorigin_isPU = jetorigin_isPU[jet];
                unpackedTree.jetorigin_isLLP_RAD= jetorigin_isLLP_RAD[jet];
                unpackedTree.jetorigin_isLLP_MU= jetorigin_isLLP_MU[jet];
                unpackedTree.jetorigin_isLLP_E= jetorigin_isLLP_E[jet];
                unpackedTree.jetorigin_isLLP_Q= jetorigin_isLLP_Q[jet];
                unpackedTree.jetorigin_isLLP_QMU= jetorigin_isLLP_QMU[jet];
                unpackedTree.jetorigin_isLLP_QE= jetorigin_isLLP_QE[jet];
                unpackedTree.jetorigin_isLLP_QQ= jetorigin_isLLP_QQ[jet];
                unpackedTree.jetorigin_isLLP_QQMU= jetorigin_isLLP_QQMU[jet];
                unpackedTree.jetorigin_isLLP_QQE= jetorigin_isLLP_QQE[jet];
                unpackedTree.jetorigin_isLLP_B= jetorigin_isLLP_B[jet];
                unpackedTree.jetorigin_isLLP_BMU= jetorigin_isLLP_BMU[jet];
                unpackedTree.jetorigin_isLLP_BE= jetorigin_isLLP_BE[jet];
                unpackedTree.jetorigin_isLLP_BB= jetorigin_isLLP_BB[jet];
                unpackedTree.jetorigin_isLLP_BBMU= jetorigin_isLLP_BBMU[jet];
                unpackedTree.jetorigin_isLLP_BBE= jetorigin_isLLP_BBE[jet];		
            }
            else
            {
                unpackedTree.isData = isData;
                unpackedTree.xsecweight = xsecweight;
                unpackedTree.processId = processId;
            }
            
            
            unpackedTree.global_pt = global_pt[jet];
            unpackedTree.global_eta = global_eta[jet];
            unpackedTree.global_mass= global_mass[jet];
            unpackedTree.global_n60= global_n60[jet];
            unpackedTree.global_n90= global_n90[jet];
            unpackedTree.global_chargedEmEnergyFraction= global_chargedEmEnergyFraction[jet];
            unpackedTree.global_chargedHadronEnergyFraction= global_chargedHadronEnergyFraction[jet];
            unpackedTree.global_chargedMuEnergyFraction= global_chargedMuEnergyFraction[jet];
            unpackedTree.global_electronEnergyFraction= global_electronEnergyFraction[jet];

            unpackedTree.global_tau1= global_tau1[jet];
            unpackedTree.global_tau2= global_tau2[jet];
            unpackedTree.global_tau3= global_tau3[jet];
    
            unpackedTree.global_relMassDropMassAK= global_relMassDropMassAK[jet];
            unpackedTree.global_relMassDropMassCA= global_relMassDropMassCA[jet];
            unpackedTree.global_relSoftDropMassAK= global_relSoftDropMassAK[jet];
            unpackedTree.global_relSoftDropMassCA= global_relSoftDropMassCA[jet];
    
            unpackedTree.global_thrust= global_thrust[jet];
            unpackedTree.global_sphericity= global_sphericity[jet];
            unpackedTree.global_circularity= global_circularity[jet];
            unpackedTree.global_isotropy= global_isotropy[jet];
            unpackedTree.global_eventShapeC= global_eventShapeC[jet];
            unpackedTree.global_eventShapeD= global_eventShapeD[jet];
            
            
            unpackedTree.csv_trackSumJetEtRatio = csv_trackSumJetEtRatio[jet];
            unpackedTree.csv_trackSumJetDeltaR = csv_trackSumJetDeltaR[jet];
            unpackedTree.csv_vertexCategory = csv_vertexCategory[jet];
            unpackedTree.csv_trackSip2dValAboveCharm = csv_trackSip2dValAboveCharm[jet];
            unpackedTree.csv_trackSip2dSigAboveCharm = csv_trackSip2dSigAboveCharm[jet];
            unpackedTree.csv_trackSip3dValAboveCharm = csv_trackSip3dValAboveCharm[jet];
            unpackedTree.csv_trackSip3dSigAboveCharm = csv_trackSip3dSigAboveCharm[jet];
            unpackedTree.csv_jetNSelectedTracks = csv_jetNSelectedTracks[jet];
            unpackedTree.csv_jetNTracksEtaRel = csv_jetNTracksEtaRel[jet];
            unpackedTree.csv_trackSip3dSigAboveCharm = csv_trackSip3dSigAboveCharm[jet];
            unpackedTree.csv_jetNSelectedTracks = csv_jetNSelectedTracks[jet];
            unpackedTree.csv_jetNTracksEtaRel = csv_jetNTracksEtaRel[jet];

            
            int cpf_offset = 0;
            for (size_t i = 0; i < jet; ++i)
            {
                cpf_offset += length_cpf[i];
            }
            
            int ncpf = std::min<int>(UnpackedTree::maxEntries_cpf,length_cpf[jet]);
            unpackedTree.ncpf = ncpf;
            
            for (int i = 0; i < ncpf; ++i)
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
                unpackedTree.cpf_deta[i] = cpf_deta[cpf_offset+i];
                unpackedTree.cpf_dphi[i] = cpf_dphi[cpf_offset+i];
                unpackedTree.cpf_drminsv[i] = cpf_drminsv[cpf_offset+i];
                unpackedTree.cpf_vertex_association[i] = cpf_vertex_association[cpf_offset+i];
                unpackedTree.cpf_puppi_weight[i] = cpf_puppi_weight[cpf_offset+i];
                unpackedTree.cpf_fromPV[i] = cpf_fromPV[cpf_offset+i];
                unpackedTree.cpf_track_chi2[i] = cpf_track_chi2[cpf_offset+i];
                unpackedTree.cpf_track_ndof[i] = cpf_track_ndof[cpf_offset+i];
                unpackedTree.cpf_track_quality[i] = cpf_track_quality[cpf_offset+i];
                unpackedTree.cpf_relmassdrop[i] = cpf_relmassdrop[cpf_offset+i];
                
                unpackedTree.cpf_matchedSV[i] = cpf_matchedSV[cpf_offset+i];
                unpackedTree.cpf_matchedMuon[i] = cpf_matchedMuon[cpf_offset+i];
                unpackedTree.cpf_matchedElectron[i] = cpf_matchedElectron[cpf_offset+i];

		unpackedTree.cpf_track_numberOfValidPixelHits[i] = cpf_track_numberOfValidPixelHits[cpf_offset+i] ; 
		unpackedTree.cpf_track_pixelLayersWithMeasurement[i] = cpf_track_pixelLayersWithMeasurement[cpf_offset+i]; 
		unpackedTree.cpf_track_numberOfValidStripHits[i] = cpf_track_numberOfValidStripHits[cpf_offset+i] ; 
		unpackedTree.cpf_track_stripLayersWithMeasurement[i] = cpf_track_stripLayersWithMeasurement[cpf_offset+i] ;
            }
	
            
            
            int npf_offset = 0;
            for (size_t i = 0; i < jet; ++i)
            {
                npf_offset += length_npf[i];
            }
            int nnpf = std::min<int>(UnpackedTree::maxEntries_npf,length_npf[jet]);
            unpackedTree.nnpf = nnpf;
            
            for (int i = 0; i < nnpf; ++i)
            {
                unpackedTree.npf_ptrel[i] = npf_ptrel[npf_offset+i];
                unpackedTree.npf_deta[i] = npf_deta[npf_offset+i];
                unpackedTree.npf_dphi[i] = npf_dphi[npf_offset+i];
                unpackedTree.npf_deltaR[i] = npf_deltaR[npf_offset+i];
                unpackedTree.npf_isGamma[i] = npf_isGamma[npf_offset+i];
                unpackedTree.npf_hcal_fraction[i] = npf_hcal_fraction[npf_offset+i];
                unpackedTree.npf_drminsv[i] = npf_drminsv[npf_offset+i];
                unpackedTree.npf_puppi_weight[i] = npf_puppi_weight[npf_offset+i];
                unpackedTree.npf_relmassdrop[i] = npf_relmassdrop[npf_offset+i];
            }

            int sv_offset = 0;
            for (size_t i = 0; i < jet; ++i)
            {
                sv_offset += length_sv[i];
            }
            
            int nsv = std::min<int>(UnpackedTree::maxEntries_sv,length_sv[jet]);
            unpackedTree.nsv = nsv;
            
            for (int i = 0; i < nsv; ++i)
            {
                unpackedTree.sv_ptrel[i] = sv_ptrel[sv_offset+i];
                unpackedTree.sv_deta[i] = sv_deta[sv_offset+i];
                unpackedTree.sv_dphi[i] = sv_dphi[sv_offset+i];
                unpackedTree.sv_mass[i] = sv_mass[sv_offset+i];
                unpackedTree.sv_deltaR[i] = sv_deltaR[sv_offset+i];
                unpackedTree.sv_ntracks[i] = sv_ntracks[sv_offset+i];
                unpackedTree.sv_chi2[i] = sv_chi2[sv_offset+i];
                unpackedTree.sv_ndof[i] = sv_ndof[sv_offset+i];
                unpackedTree.sv_dxy[i] = sv_dxy[sv_offset+i];
                unpackedTree.sv_dxysig[i] = sv_dxysig[sv_offset+i];
                unpackedTree.sv_d3d[i] = sv_d3d[sv_offset+i];
                unpackedTree.sv_d3dsig[i] = sv_d3dsig[sv_offset+i];
                unpackedTree.sv_costhetasvpv[i] = sv_costhetasvpv[sv_offset+i];
                unpackedTree.sv_enratio[i] = sv_enratio[sv_offset+i];
            }
            
            
            int muon_offset = 0;
            for (size_t i = 0; i < jet; ++i)
            {
                muon_offset += length_muon[i];
            }
            
            int nmuon = std::min<int>(UnpackedTree::maxEntries_muon,length_muon[jet]);
            unpackedTree.nmuon = nmuon;
            
            for (int i = 0; i < nmuon; ++i)
            {
                unpackedTree.muon_isGlobal[i] = muon_isGlobal[muon_offset+i];
                unpackedTree.muon_isTight[i] = muon_isTight[muon_offset+i];
                unpackedTree.muon_isMedium[i] = muon_isMedium[muon_offset+i]; 
                unpackedTree.muon_isLoose[i] = muon_isLoose[muon_offset+i];
                unpackedTree.muon_isStandAlone[i] = muon_isStandAlone[muon_offset+i];

                unpackedTree.muon_ptrel[i] = muon_ptrel[muon_offset+i];
                unpackedTree.muon_EtaRel[i] = muon_EtaRel[muon_offset+i];
                unpackedTree.muon_dphi[i] = muon_dphi[muon_offset+i];
                unpackedTree.muon_deta[i] = muon_deta[muon_offset+i];
                unpackedTree.muon_charge[i] = muon_charge[muon_offset+i];
                unpackedTree.muon_energy[i] = muon_energy[muon_offset+i];
                unpackedTree.muon_jetDeltaR[i] = muon_jetDeltaR[muon_offset+i]; 
                unpackedTree.muon_numberOfMatchedStations[i] = muon_numberOfMatchedStations[muon_offset+i];

                unpackedTree.muon_2dIp[i] = muon_2dIp[muon_offset+i];
                unpackedTree.muon_2dIpSig[i] = muon_2dIpSig[muon_offset+i];
                unpackedTree.muon_3dIp[i] = muon_3dIp[muon_offset+i];
                unpackedTree.muon_3dIpSig[i] = muon_3dIpSig[muon_offset+i];

                unpackedTree.muon_dxy[i] = muon_dxy[muon_offset+i];
                unpackedTree.muon_dxyError[i] = muon_dxyError[muon_offset+i];
                unpackedTree.muon_dxySig[i] = muon_dxySig[muon_offset+i]; 
                unpackedTree.muon_dz[i] = muon_dz[muon_offset+i];
                unpackedTree.muon_dzError[i] = muon_dzError[muon_offset+i];
                unpackedTree.muon_numberOfValidPixelHits[i] = muon_numberOfValidPixelHits[muon_offset+i]; 
                unpackedTree.muon_numberOfpixelLayersWithMeasurement[i] = muon_numberOfpixelLayersWithMeasurement[muon_offset+i];

                unpackedTree.muon_chi2[i] = muon_chi2[muon_offset+i];
                unpackedTree.muon_ndof[i] = muon_ndof[muon_offset+i];

                unpackedTree.muon_caloIso[i] = muon_caloIso[muon_offset+i];
                unpackedTree.muon_ecalIso[i] = muon_ecalIso[muon_offset+i];
                unpackedTree.muon_hcalIso[i] = muon_hcalIso[muon_offset+i];

                unpackedTree.muon_sumPfChHadronPt[i] = muon_sumPfChHadronPt[muon_offset+i];
                unpackedTree.muon_sumPfNeuHadronEt[i] = muon_sumPfNeuHadronEt[muon_offset+i];
                unpackedTree.muon_Pfpileup[i] = muon_Pfpileup[muon_offset+i];
                unpackedTree.muon_sumPfPhotonEt[i] = muon_sumPfPhotonEt[muon_offset+i];

                unpackedTree.muon_sumPfChHadronPt03[i] = muon_sumPfChHadronPt03[muon_offset+i];
                unpackedTree.muon_sumPfNeuHadronEt03[i] = muon_sumPfNeuHadronEt03[muon_offset+i];
                unpackedTree.muon_Pfpileup03[i] = muon_Pfpileup03[muon_offset+i];
                unpackedTree.muon_sumPfPhotonEt03[i] = muon_sumPfPhotonEt03[muon_offset+i];

                unpackedTree.muon_sumChHadronPt[i] = muon_sumChHadronPt[muon_offset+i];
                unpackedTree.muon_sumNeuHadronEt[i] = muon_sumNeuHadronEt[muon_offset+i];
                unpackedTree.muon_pileup[i] = muon_pileup[muon_offset+i];
                unpackedTree.muon_sumPhotonEt[i] = muon_sumPhotonEt[muon_offset+i];

                unpackedTree.muon_timeAtIpInOut[i] = muon_timeAtIpInOut[muon_offset+i];
                unpackedTree.muon_timeAtIpInOutErr[i] = muon_timeAtIpInOutErr[muon_offset+i];
                unpackedTree.muon_timeAtIpOutIn[i] = muon_timeAtIpOutIn[muon_offset+i];

	    }

            int electron_offset = 0;
            for (size_t i = 0; i < jet; ++i)
            {
                electron_offset += length_electron[i];
            }
            
            int nelectron = std::min<int>(UnpackedTree::maxEntries_electron,length_electron[jet]);
            unpackedTree.nelectron = nelectron;

            for (int i = 0; i < nelectron; ++i)
            {

       	        unpackedTree.electron_jetIdx[i] = electron_jetIdx[electron_offset+i];
                unpackedTree.electron_ptrel[i] = electron_ptrel[electron_offset+i]; 
                unpackedTree.electron_jetDeltaR[i] =electron_jetDeltaR[electron_offset+i];  
                unpackedTree.electron_deta[i] = electron_deta[electron_offset+i]; 
                unpackedTree.electron_dphi[i] = electron_dphi[electron_offset+i]; 
                unpackedTree.electron_charge[i] =electron_charge[electron_offset+i];  
                unpackedTree.electron_energy[i] = electron_energy[electron_offset+i]; 
                unpackedTree.electron_EtFromCaloEn[i] = electron_EtFromCaloEn[electron_offset+i]; 
                unpackedTree.electron_isEB[i] = electron_isEB[electron_offset+i];  
                unpackedTree.electron_isEE[i] = electron_isEE[electron_offset+i];  
                unpackedTree.electron_ecalEnergy[i] = electron_ecalEnergy[electron_offset+i];  
                unpackedTree.electron_isPassConversionVeto[i] = electron_isPassConversionVeto[electron_offset+i]; 
                unpackedTree.electron_convDist[i] = electron_convDist[electron_offset+i];  
    	        unpackedTree.electron_convFlags[i] = electron_convFlags[electron_offset+i];  
                unpackedTree.electron_convRadius[i] = electron_convRadius[electron_offset+i];  
                unpackedTree.electron_hadronicOverEm[i] = electron_hadronicOverEm[electron_offset+i]; 
                unpackedTree.electron_ecalDrivenSeed[i] = electron_ecalDrivenSeed[electron_offset+i]; 

    // superCluster block

                unpackedTree.electron_SC_energy[i] = electron_SC_energy[electron_offset+i];  
                unpackedTree.electron_SC_deta[i] = electron_SC_deta[electron_offset+i];  
                unpackedTree.electron_SC_dphi[i] = electron_SC_dphi[electron_offset+i]; 
                unpackedTree.electron_SC_et[i] = electron_SC_et[electron_offset+i]; 
                unpackedTree.electron_SC_eSuperClusterOverP[i] = electron_SC_eSuperClusterOverP[electron_offset+i];  
                unpackedTree.electron_scE1x5Overe5x5[i] = electron_scE1x5Overe5x5[electron_offset+i];  
                unpackedTree.electron_scE2x5MaxOvere5x5[i] = electron_scE2x5MaxOvere5x5[electron_offset+i];  
                unpackedTree.electron_scE5x5[i] = electron_scE5x5[electron_offset+i];  
                unpackedTree.electron_scE5x5Rel [i] = electron_scE5x5Rel[electron_offset+i];  
                unpackedTree.electron_scPixCharge [i] = electron_scPixCharge[electron_offset+i];  
                unpackedTree.electron_scSigmaEtaEta [i] = electron_scSigmaEtaEta[electron_offset+i]; 
                unpackedTree.electron_scSigmaIEtaIEta [i] = electron_scSigmaIEtaIEta[electron_offset+i];   
                unpackedTree.electron_superClusterFbrem [i] = electron_superClusterFbrem[electron_offset+i];  

                unpackedTree.electron_2dIP[i] = electron_2dIP[electron_offset+i];  
                unpackedTree.electron_2dIPSig[i] =electron_2dIPSig[electron_offset+i]; 
                unpackedTree.electron_3dIP[i] = electron_3dIP[electron_offset+i];  
                unpackedTree.electron_3dIPSig[i] =electron_3dIPSig[electron_offset+i];  
                unpackedTree.electron_eSeedClusterOverP[i] = electron_eSeedClusterOverP[electron_offset+i]; 
                unpackedTree.electron_eSeedClusterOverPout[i] =electron_eSeedClusterOverPout[electron_offset+i]; 
                unpackedTree.electron_eSuperClusterOverP[i] = electron_eSuperClusterOverP[electron_offset+i]; 
                unpackedTree.electron_eTopOvere5x5[i] = electron_eTopOvere5x5[electron_offset+i];  

                unpackedTree.electron_deltaEtaEleClusterTrackAtCalo [i] =electron_deltaEtaEleClusterTrackAtCalo [electron_offset+i];  
                unpackedTree.electron_deltaEtaSeedClusterTrackAtCalo [i] = electron_deltaEtaSeedClusterTrackAtCalo[electron_offset+i]; 
                unpackedTree.electron_deltaPhiSeedClusterTrackAtCalo [i] = electron_deltaPhiSeedClusterTrackAtCalo[electron_offset+i];  
                unpackedTree.electron_deltaEtaSeedClusterTrackAtVtx [i] = electron_deltaEtaSeedClusterTrackAtVtx[electron_offset+i];  
                unpackedTree.electron_deltaEtaSuperClusterTrackAtVtx [i] = electron_deltaEtaSuperClusterTrackAtVtx[electron_offset+i]; 
                unpackedTree.electron_deltaPhiEleClusterTrackAtCalo [i] = electron_deltaPhiEleClusterTrackAtCalo[electron_offset+i];  
                unpackedTree.electron_deltaPhiSuperClusterTrackAtVtx [i] = electron_deltaPhiSuperClusterTrackAtVtx[electron_offset+i]; 
                unpackedTree.electron_sCseedEta [i] = electron_sCseedEta[electron_offset+i];   

    //     unpackedTree.electron gsf variables. 
                unpackedTree.electron_EtaRel [i] = electron_EtaRel[electron_offset+i];  
                unpackedTree.electron_dxy [i] = electron_dxy[electron_offset+i];  
                unpackedTree.electron_dz [i] = electron_dz[electron_offset+i]; 
                unpackedTree.electron_nbOfMissingHits [i] = electron_nbOfMissingHits[electron_offset+i];  
                unpackedTree.electron_gsfCharge [i] = electron_gsfCharge[electron_offset+i]; 


                unpackedTree.electron_numberOfBrems [i] = electron_numberOfBrems[electron_offset+i]; 
                unpackedTree.electron_trackFbrem [i] = electron_trackFbrem[electron_offset+i];  
                unpackedTree.electron_fbrem [i] = electron_fbrem[electron_offset+i];  
                unpackedTree.electron_e5x5 [i] = electron_e5x5[electron_offset+i];  
                unpackedTree.electron_e5x5Rel [i] =electron_e5x5Rel[electron_offset+i];  
                unpackedTree.electron_e1x5Overe5x5 [i] = electron_e1x5Overe5x5[electron_offset+i];  
                unpackedTree.electron_e2x5MaxOvere5x5[i] = electron_e2x5MaxOvere5x5[electron_offset+i]; 

    // 5*5 cells cluster 
    //

                unpackedTree.electron_full5x5_e5x5 [i] = electron_full5x5_e5x5[electron_offset+i]; 
                unpackedTree.electron_full5x5_e5x5Rel [i] = electron_full5x5_e5x5Rel[electron_offset+i];  
                unpackedTree.electron_full5x5_sigmaIetaIeta [i] = electron_full5x5_sigmaIetaIeta[electron_offset+i]; 
                unpackedTree.electron_full5x5_e1x5Overe5x5 [i] = electron_full5x5_e1x5Overe5x5[electron_offset+i]; 
                unpackedTree.electron_full5x5_e2x5BottomOvere5x5 [i] = electron_full5x5_e2x5BottomOvere5x5[electron_offset+i]; 
                unpackedTree.electron_full5x5_e2x5LeftOvere5x5 [i] = electron_full5x5_e2x5LeftOvere5x5[electron_offset+i]; 
                unpackedTree.electron_full5x5_e2x5MaxOvere5x5 [i] = electron_full5x5_e2x5MaxOvere5x5[electron_offset+i]; 
                unpackedTree.electron_full5x5_e2x5RightOvere5x5 [i] = electron_full5x5_e2x5RightOvere5x5[electron_offset+i]; 
                unpackedTree.electron_full5x5_e2x5TopOvere5x5 [i] = electron_full5x5_e2x5TopOvere5x5[electron_offset+i]; 



                unpackedTree.electron_full5x5_eBottomOvere5x5[i] = electron_full5x5_eBottomOvere5x5[electron_offset+i]; 
                unpackedTree.electron_full5x5_eLeftOvere5x5[i] = electron_full5x5_eLeftOvere5x5[electron_offset+i]; 
                unpackedTree.electron_full5x5_eRightOvere5x5[i] = electron_full5x5_eRightOvere5x5[electron_offset+i]; 
                unpackedTree.electron_full5x5_eTopOvere5x5[i] = electron_full5x5_eTopOvere5x5[electron_offset+i]; 
                unpackedTree.electron_full5x5_hcalDepth1OverEcal [i] = electron_full5x5_hcalDepth1OverEcal[electron_offset+i]; 
                unpackedTree.electron_full5x5_hcalDepth1OverEcalBc [i] = electron_full5x5_hcalDepth1OverEcalBc[electron_offset+i]; 
                unpackedTree.electron_full5x5_hcalDepth2OverEcal[i] = electron_full5x5_hcalDepth2OverEcal[electron_offset+i]; 
                unpackedTree.electron_full5x5_hcalDepth2OverEcalBc [i] = electron_full5x5_hcalDepth2OverEcalBc[electron_offset+i]; 
                unpackedTree.electron_full5x5_hcalOverEcal [i] = electron_full5x5_hcalOverEcal[electron_offset+i]; 
                unpackedTree.electron_full5x5_hcalOverEcalBc[i] = electron_full5x5_hcalOverEcalBc[electron_offset+i];    
                unpackedTree.electron_full5x5_r9 [i] = electron_full5x5_r9[electron_offset+i]; 



    // Isolation block
                unpackedTree.electron_neutralHadronIso[i] = electron_neutralHadronIso[electron_offset+i];  
                unpackedTree.electron_particleIso [i] = electron_particleIso[electron_offset+i];  
                unpackedTree.electron_photonIso[i] = electron_photonIso[electron_offset+i]; 
                unpackedTree.electron_puChargedHadronIso[i] = electron_puChargedHadronIso[electron_offset+i];  
                unpackedTree.electron_trackIso [i] = electron_trackIso[electron_offset+i];   
                unpackedTree.electron_hcalDepth1OverEcal [i] = electron_hcalDepth1OverEcal[electron_offset+i];  
                unpackedTree.electron_hcalDepth2OverEcal [i] = electron_hcalDepth2OverEcal[electron_offset+i];  
                unpackedTree.electron_ecalPFClusterIso [i] = electron_ecalPFClusterIso[electron_offset+i]; 
                unpackedTree.electron_hcalPFClusterIso [i] = electron_hcalPFClusterIso[electron_offset+i];   
                unpackedTree.electron_dr03TkSumPt [i] = electron_dr03TkSumPt[electron_offset+i];  

                unpackedTree.electron_dr03EcalRecHitSumEt [i] = electron_dr03EcalRecHitSumEt[electron_offset+i];  
                unpackedTree.electron_dr03HcalDepth1TowerSumEt [i] = electron_dr03HcalDepth1TowerSumEt[electron_offset+i];   
                unpackedTree.electron_dr03HcalDepth1TowerSumEtBc [i] = electron_dr03HcalDepth1TowerSumEtBc[electron_offset+i];  
                unpackedTree.electron_dr03HcalDepth2TowerSumEt [i] = electron_dr03HcalDepth2TowerSumEt[electron_offset+i];  
                unpackedTree.electron_dr03HcalDepth2TowerSumEtBc [i] = electron_dr03HcalDepth2TowerSumEtBc[electron_offset+i];  
                unpackedTree.electron_pfSumPhotonEt [i] = electron_pfSumPhotonEt[electron_offset+i];  
                unpackedTree.electron_pfSumChargedHadronPt [i] = electron_pfSumChargedHadronPt[electron_offset+i];  
                unpackedTree.electron_pfSumNeutralHadronEt [i] = electron_pfSumNeutralHadronEt[electron_offset+i];  
                unpackedTree.electron_pfSumPUPt [i] = electron_pfSumPUPt[electron_offset+i]; 

                unpackedTree.electron_dr04EcalRecHitSumEt [i] = electron_dr04EcalRecHitSumEt[electron_offset+i];   
                unpackedTree.electron_dr04HcalDepth1TowerSumEt [i] = electron_dr04HcalDepth1TowerSumEt[electron_offset+i];   
                unpackedTree.electron_dr04HcalDepth1TowerSumEtBc [i] = electron_dr04HcalDepth1TowerSumEtBc[electron_offset+i]; 
                unpackedTree.electron_dr04HcalDepth2TowerSumEt [i] = electron_dr04HcalDepth2TowerSumEt[electron_offset+i];  
                unpackedTree.electron_dr04HcalDepth2TowerSumEtBc  [i] = electron_dr04HcalDepth2TowerSumEtBc[electron_offset+i]; 
                unpackedTree.electron_dr04HcalTowerSumEt  [i] = electron_dr04HcalTowerSumEt[electron_offset+i]; 
                unpackedTree.electron_dr04HcalTowerSumEtBc  [i] = electron_dr04HcalTowerSumEtBc[electron_offset+i]; 
		
		
            }

            
            unpackedTree.fill();
            return true;
        }
       
};

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

unsigned int calcHash(unsigned int value)
{
    unsigned int hash = ((value >> 16) ^ value) * 0x45d9f3b;
    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = (hash >> 16) ^ hash;
    return hash;
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
    std::cout<<"current split: "<<iSplit<<"/"<<nSplit<<std::endl;
    if (iSplit>=nSplit)
    {
        std::cout<<"Error: Current split number (-b) needs to be smaller than total split (-s) number!"<<std::endl;
        return 1;
    }
    
    bool addTruth = parser.get<bool>("t");
    std::cout<<"add truth from jetorigin: "<<(addTruth ? "true" : "false")<<std::endl;
    
    std::vector<std::unique_ptr<NanoXTree>> trees;
    std::cout<<"Input files: "<<std::endl;
    
    std::vector<int> entries;
    int total_entries = 0;
    
    std::vector<std::vector<std::string>> inputFileNames;
    std::vector<std::vector<std::string>> selectors;
    std::vector<std::vector<std::string>> setters;
    std::vector<int> caps;
    
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
            setters.push_back(std::vector<std::string>{});
            caps.push_back(-1);
        }
        else if(ends_with(s,".txt"))
        {
            std::ifstream input(s);
            std::vector<std::string> files;
            std::vector<std::string> select;
            std::vector<std::string> setter;
            int cap = -1;
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
                    else if (begins_with(line,"#cap"))
                    {
                        cap = atoi(std::string(line.begin()+4,line.end()).c_str());
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
            caps.push_back(cap);
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
        //int nfiles = 0;
        int totalEntries = 0;
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
            totalEntries += nEvents;
            std::cout<<"   "<<inputFileName<<", nEvents="<<nEvents<<std::endl;
            file->Close();
            chain->AddFile(inputFileName.c_str());
            
            if (caps[i]>0 and totalEntries>caps[i])
            {
                std::cout<<"   "<<inputFileName<<"number of "<<caps[i]<<" events reached"<<std::endl;
                break;
            }
            //nfiles+=1;
            //if (nfiles>1) break;
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
    
    std::cout<<"Number of independent inputs: "<<trees.size()<<std::endl;
    std::cout<<"Total number of events: "<<total_entries<<std::endl;
    std::vector<std::unique_ptr<UnpackedTree>> unpackedTreesTrain;
    std::vector<std::vector<int>> eventsPerClassPerFileTrain(28,std::vector<int>(nOutputs,0));
    
    std::vector<std::unique_ptr<UnpackedTree>> unpackedTreesTest;
    std::vector<std::vector<int>> eventsPerClassPerFileTest(28,std::vector<int>(nOutputs,0));

    for (unsigned int i = 0; i < nOutputs; ++i)
    {
        unpackedTreesTrain.emplace_back(std::unique_ptr<UnpackedTree>(
            new UnpackedTree(outputPrefix+"_train"+std::to_string(iSplit+1)+"_"+std::to_string(i+1)+".root",addTruth
        )));

        unpackedTreesTest.emplace_back(std::unique_ptr<UnpackedTree>(
            new UnpackedTree(outputPrefix+"_test"+std::to_string(iSplit+1)+"_"+std::to_string(i+1)+".root",addTruth
        )));
    }
    
    int eventsInBatch = int(1.*total_entries/nSplit);
    
    std::cout<<"Batch number of events: "<<eventsInBatch<<std::endl;
    
    //offset reading for each input tree
    for (size_t itree = 0; itree < trees.size(); ++itree)
    {
        trees[itree]->getEvent(int(1.*iSplit*trees[itree]->entries()/nSplit),true);
    }
    
    std::vector<int> readEvents(entries.size(),0);
    for (int ientry = 0; ientry<eventsInBatch; ++ientry)
    {
        if (ientry%10000==0)
        {
            std::cout<<"Processing ... "<<100.*ientry/eventsInBatch<<std::endl;
        }
        
        
         
        //choose input file pseudo-randomly
        long hash = calcHash(47*ientry+iSplit*23);
        long hashEntries = (hash+hash/eventsInBatch)%eventsInBatch;
        
        
        int sum_entries = 0;
        int ifile = 0;
        
        for (;ifile<(entries.size()-1); ++ifile)
        {
            sum_entries += int(1.*entries[ifile]/nSplit);
            if (hashEntries<sum_entries) break;
        }
       
        trees[ifile]->nextEvent(); //this loops back to 0 in case it was the last event
        
        readEvents[ifile]+=1;
        
        for (size_t j = 0; j < std::min<int>(20,trees[ifile]->njets()); ++j)
        {
            if (trees[ifile]->isSelected(j))
            {
                int jet_class = trees[ifile]->getJetClass(j);
                long hashTest = calcHash(11*trees[ifile]->entry()+23*j)%100;
                if (hashTest<nTestFrac)
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
        std::cout<<"infile "<<i<<": found = "<<entries[i]<<", read = "<<readEvents[i]<<"/"<<int(1.*entries[i]/nSplit)<<std::endl;
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
