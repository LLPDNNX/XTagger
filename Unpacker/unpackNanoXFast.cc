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

class Feature
{
    public:
        enum Type
        {
            Float,
            Int,
            Double,
            Bool,
            Short,
            Char
        };
        
    protected:
        std::string name_;
        Type type_;
        
    public:
        Feature(const std::string name, const Type& type = Float):
            name_(name),
            type_(type)
        {
        }
        
        inline std::string name() const
        {
            return name_;
        }
        
        inline std::string rootTypeId() const
        {
            //https://root.cern.ch/doc/master/classTTree.html
            switch (type_)
            {
                case Float: return "F";
                case Int: return "I";
                case Double: return "D";
                case Bool: return "O";
                case Short: return "S";
                case Char: return "C";
            }
        }
        
        inline Type type() const
        {
            return type_;
        }
};

class BranchData
{
    public:
        virtual void setFloat(size_t index, float value) = 0;
        virtual float getFloat(size_t index) = 0;
        template<class TYPE, size_t N> static std::shared_ptr<BranchData> makeBranch(TTree* tree, const std::string& branchName, const std::string& branchType, long bufferSize);
        template<class TYPE, size_t N> static std::shared_ptr<BranchData> branchAddress(TTree* tree, const std::string& branchName);
        
        template<size_t N> static std::shared_ptr<BranchData> makeBranch(Feature::Type type, TTree* tree, const std::string& branchName, const std::string& branchType, long bufferSize)
        {
            switch (type)
            {
                case Feature::Float: return makeBranch<float,N>(tree,branchName,branchType,bufferSize);
                case Feature::Int: return makeBranch<int,N>(tree,branchName,branchType,bufferSize);
                case Feature::Double: return makeBranch<double,N>(tree,branchName,branchType,bufferSize);
                case Feature::Bool: return makeBranch<bool,N>(tree,branchName,branchType,bufferSize);
                case Feature::Short: return makeBranch<short,N>(tree,branchName,branchType,bufferSize);
                case Feature::Char: return makeBranch<char,N>(tree,branchName,branchType,bufferSize);
            }
        }
        
        template<size_t N> static std::shared_ptr<BranchData> branchAddress(Feature::Type type, TTree* tree, const std::string& branchName)
        {
            switch (type)
            {
                case Feature::Float: return branchAddress<float,N>(tree,branchName);
                case Feature::Int: return branchAddress<int,N>(tree,branchName);
                case Feature::Double: return branchAddress<double,N>(tree,branchName);
                case Feature::Bool: return branchAddress<bool,N>(tree,branchName);
                case Feature::Short: return branchAddress<short,N>(tree,branchName);
                case Feature::Char: return branchAddress<char,N>(tree,branchName);
            }
        }
};

template<class TYPE, size_t N>
class BranchDataTmpl:
    public BranchData
{
    protected:
        TYPE buffer_[N];
    public:
    
        virtual void setFloat(size_t index, float value)
        {
            buffer_[index] = TYPE(value);
        }
        virtual float getFloat(size_t index)
        {
            return buffer_[index];
        }
        
        inline TYPE* buffer()
        {
            return buffer_;
        }
};

//scalar specialization
template<class TYPE>
class BranchDataTmpl<TYPE,0>:
    public BranchData
{
    protected:
        TYPE buffer_;
    public:
    
        virtual void setFloat(size_t index, float value)
        {
            buffer_ = TYPE(value);
        }
        virtual float getFloat(size_t index)
        {
            return buffer_;
        }
        
        inline TYPE* buffer()
        {
            return &buffer_;
        }
};



template<class TYPE, size_t N> 
std::shared_ptr<BranchData> BranchData::makeBranch(TTree* tree, const std::string& branchName, const std::string& branchType, long bufferSize)
{
    std::shared_ptr<BranchDataTmpl<TYPE,N>> branchData(new BranchDataTmpl<TYPE,N>());
    tree->Branch(branchName.c_str(),branchData->buffer(),branchType.c_str(),bufferSize);
    return branchData;
}

template<class TYPE, size_t N> 
std::shared_ptr<BranchData> BranchData::branchAddress(TTree* tree, const std::string& branchName)
{
    std::shared_ptr<BranchDataTmpl<TYPE,N>> branchData(new BranchDataTmpl<TYPE,N>());
    tree->SetBranchAddress(branchName.c_str(),branchData->buffer());
    return branchData;
}

static const std::vector<Feature> globalFeatures{
    Feature("global_mass"),
    Feature("global_area"),
    Feature("global_n60",Feature::Int),
    Feature("global_n90",Feature::Int),
    Feature("global_chargedEmEnergyFraction"),
    Feature("global_chargedHadronEnergyFraction"),
    Feature("global_chargedMuEnergyFraction"),
    Feature("global_electronEnergyFraction"),

    Feature("global_tau1"),
    Feature("global_tau2"),
    Feature("global_tau3"),
    
    Feature("global_relMassDropMassAK"),
    Feature("global_relMassDropMassCA"),
    Feature("global_relSoftDropMassAK"),
    Feature("global_relSoftDropMassCA"),
    
    Feature("global_thrust"),
    Feature("global_sphericity"),
    Feature("global_circularity"),
    Feature("global_isotropy"),
    Feature("global_eventShapeC"),
    Feature("global_eventShapeD")
};

static const std::vector<Feature> csvFeatures{ 
    Feature("csv_trackSumJetEtRatio"),
    Feature("csv_trackSumJetDeltaR"),
    Feature("csv_vertexCategory"),
    Feature("csv_trackSip2dValAboveCharm"),
    Feature("csv_trackSip2dSigAboveCharm"),
    Feature("csv_trackSip3dValAboveCharm"),
    Feature("csv_trackSip3dSigAboveCharm"),
    Feature("csv_jetNSelectedTracks",Feature::Int),
    Feature("csv_jetNTracksEtaRel",Feature::Int)
};


static const std::vector<Feature> cpfFeatures{
    Feature("cpf_trackEtaRel"),
    Feature("cpf_trackPtRel"),
    Feature("cpf_trackPPar"),
    Feature("cpf_trackDeltaR"),
    Feature("cpf_trackPParRatio"),
    Feature("cpf_trackPtRatio"),
    Feature("cpf_trackSip2dVal"),
    Feature("cpf_trackSip2dSig"),
    Feature("cpf_trackSip3dVal"),
    Feature("cpf_trackSip3dSig"),
    Feature("cpf_trackJetDistVal"),
    Feature("cpf_trackJetDistSig"),
    Feature("cpf_ptrel"),
    Feature("cpf_drminsv"),
    Feature("cpf_vertex_association"),
    Feature("cpf_fromPV"),
    Feature("cpf_puppi_weight"),
    Feature("cpf_track_chi2"),
    Feature("cpf_track_quality"),
    Feature("cpf_track_ndof", Feature::Int),
    Feature("cpf_matchedMuon", Feature::Int),
    Feature("cpf_matchedElectron", Feature::Int),
    Feature("cpf_matchedSV", Feature::Int),
    Feature("cpf_numberOfValidPixelHits", Feature::Int),
    Feature("cpf_pixelLayersWithMeasurement",Feature::Int),
    Feature("cpf_numberOfValidStripHits" , Feature::Int),
    Feature("cpf_stripLayersWithMeasurement" , Feature::Int),
    Feature("cpf_relmassdrop"),
    Feature("cpf_dzMin")
    //Feature("cpf_deta"),
    //Feature("cpf_dphi")
};

static const std::vector<Feature> npfFeatures{
    Feature("npf_ptrel"),
    Feature("npf_deta"),
    Feature("npf_dphi"),
    Feature("npf_deltaR"),
    Feature("npf_isGamma"),
    Feature("npf_hcal_fraction"),
    Feature("npf_drminsv"),
    Feature("npf_puppi_weight"),
    Feature("npf_relmassdrop")
};


static const std::vector<Feature> svFeatures{
    Feature("sv_ptrel"),
    Feature("sv_deta"),
    Feature("sv_dphi"),
    Feature("sv_deltaR"),
    Feature("sv_mass"),
    Feature("sv_ntracks"),
    Feature("sv_chi2"),
    Feature("sv_ndof",Feature::Int),
    Feature("sv_dxy"),
    Feature("sv_dxysig"),
    Feature("sv_d3d"),
    Feature("sv_d3dsig"),
    Feature("sv_costhetasvpv"),
    Feature("sv_enratio")
};

static const std::vector<Feature> muonFeatures{
    Feature("muon_isGlobal",Feature::Int),
    Feature("muon_isTight",Feature::Int),
    Feature("muon_isMedium",Feature::Int),
    Feature("muon_isLoose",Feature::Int),
    Feature("muon_isStandAlone",Feature::Int),

    Feature("muon_ptrel"),
    Feature("muon_EtaRel"),
    Feature("muon_dphi"),
    Feature("muon_deta"),
    Feature("muon_charge"),
    Feature("muon_energy"),
    Feature("muon_jetDeltaR"),
    Feature("muon_numberOfMatchedStations"),

    Feature("muon_2dIP"),
    Feature("muon_2dIPSig"),
    Feature("muon_3dIP"),
    Feature("muon_3dIPSig"),

    Feature("muon_dxy"),
    Feature("muon_dxyError"),
    Feature("muon_dxySig"),
    Feature("muon_dz"),
    Feature("muon_dzError"),
    Feature("muon_numberOfValidPixelHits",Feature::Int),
    Feature("muon_numberOfpixelLayersWithMeasurement",Feature::Int),
    Feature("muon_numberOfstripLayersWithMeasurement",Feature::Int),

    Feature("muon_chi2"),
    Feature("muon_ndof",Feature::Int),

    Feature("muon_caloIso"),
    Feature("muon_ecalIso"),
    Feature("muon_hcalIso"),

    Feature("muon_sumPfChHadronPt"),
    Feature("muon_sumPfNeuHadronEt"),
    Feature("muon_Pfpileup"),
    Feature("muon_sumPfPhotonEt"),

    Feature("muon_sumPfChHadronPt03"),
    Feature("muon_sumPfNeuHadronEt03"),
    Feature("muon_Pfpileup03"),
    Feature("muon_sumPfPhotonEt03"),

    Feature("muon_timeAtIpInOut"),
    Feature("muon_timeAtIpInOutErr"),
    Feature("muon_timeAtIpOutIn")
};

static const std::vector<Feature> electronFeatures{
    Feature("electron_ptrel"),
    Feature("electron_jetDeltaR"),
    Feature("electron_deta"),
    Feature("electron_dphi"),
    Feature("electron_charge"),
    Feature("electron_energy"),
    Feature("electron_EtFromCaloEn"),
    Feature("electron_isEB"),
    Feature("electron_isEE"),
    Feature("electron_ecalEnergy"),
    Feature("electron_isPassConversionVeto"),
    Feature("electron_convDist"),
    Feature("electron_convFlags",Feature::Int),

    Feature("electron_convRadius"),
    Feature("electron_hadronicOverEm"),
    Feature("electron_ecalDrivenSeed"),


    Feature("electron_SC_energy"),
    Feature("electron_SC_deta"),
    Feature("electron_SC_dphi"),
    Feature("electron_SC_et"),
    Feature("electron_SC_eSuperClusterOverP"),
    Feature("electron_scPixCharge"),
    Feature("electron_sigmaEtaEta"),
    Feature("electron_sigmaIetaIeta"),
    Feature("electron_sigmaIphiIphi"),
    Feature("electron_r9"),
    Feature("electron_superClusterFbrem"),

    Feature("electron_2dIP"),
    Feature("electron_2dIPSig"),
    Feature("electron_3dIP"),
    Feature("electron_3dIPSig"),
    Feature("electron_eSeedClusterOverP"),
    Feature("electron_eSeedClusterOverPout"),
    Feature("electron_eSuperClusterOverP"),

    Feature("electron_deltaEtaEleClusterTrackAtCalo"),
    Feature("electron_deltaEtaSeedClusterTrackAtCalo"),
    Feature("electron_deltaPhiSeedClusterTrackAtCalo"),
    Feature("electron_deltaEtaSeedClusterTrackAtVtx"),
    Feature("electron_deltaEtaSuperClusterTrackAtVtx"),
    Feature("electron_deltaPhiEleClusterTrackAtCalo"),
    Feature("electron_deltaPhiSuperClusterTrackAtVtx"),
    Feature("electron_sCseedEta"),

    Feature("electron_EtaRel"),
    Feature("electron_dxy"),
    Feature("electron_dz"),
    Feature("electron_nbOfMissingHits"),
    Feature("electron_gsfCharge"),
    Feature("electron_ndof",Feature::Int),
    Feature("electron_chi2"),

    Feature("electron_numberOfBrems",Feature::Int),
    Feature("electron_fbrem"),

    Feature("electron_e5x5"),
    Feature("electron_e5x5Rel"),
    Feature("electron_e2x5MaxOvere5x5"),
    Feature("electron_e1x5Overe5x5"),

    Feature("electron_neutralHadronIso"),
    Feature("electron_particleIso"),
    Feature("electron_photonIso"),
    Feature("electron_puChargedHadronIso"),
    Feature("electron_trackIso"),
    Feature("electron_hcalDepth1OverEcal"),
    Feature("electron_hcalDepth2OverEcal"),
    Feature("electron_ecalPFClusterIso"),
    Feature("electron_hcalPFClusterIso"),
    Feature("electron_pfSumPhotonEt"),
    Feature("electron_pfSumChargedHadronPt"),
    Feature("electron_pfSumNeutralHadronEt"),
    Feature("electron_pfSumPUPt"),
    Feature("electron_dr04TkSumPt"),
    Feature("electron_dr04EcalRecHitSumEt"),
    Feature("electron_dr04HcalDepth1TowerSumEt"),
    Feature("electron_dr04HcalDepth1TowerSumEtBc"),
    Feature("electron_dr04HcalDepth2TowerSumEt"),
    Feature("electron_dr04HcalDepth2TowerSumEtBc"),
    Feature("electron_dr04HcalTowerSumEt"),
    Feature("electron_dr04HcalTowerSumEtBc")
};
 
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

        float global_pt;
        float global_eta;
        float global_phi;

        float isData;
        float xsecweight;
        float processId;
        
        std::vector<std::shared_ptr<BranchData>> globalBranches;
        std::vector<std::shared_ptr<BranchData>> csvBranches;
        
        unsigned int ncpf;
        std::vector<std::shared_ptr<BranchData>> cpfBranches;
        
        unsigned int nnpf;
        std::vector<std::shared_ptr<BranchData>> npfBranches;
        
        unsigned int nsv;
        std::vector<std::shared_ptr<BranchData>> svBranches;

        unsigned int nmuon;
        std::vector<std::shared_ptr<BranchData>> muonBranches;
        
        unsigned int nelectron;
        std::vector<std::shared_ptr<BranchData>> electronBranches;
        
        template<size_t N>
        std::vector<std::shared_ptr<BranchData>> makeBranches(TTree* tree, const std::vector<Feature>& features, const std::string& lengthName="") const
        {
            std::vector<std::shared_ptr<BranchData>> branches;
            for (size_t ifeature = 0; ifeature < features.size(); ++ifeature)
            {
                auto const& feature = features[ifeature];
                auto branchData = BranchData::makeBranch<N>(
                    feature.type(),
                    tree,
                    feature.name().c_str(),
                    lengthName.size()>0 ? (feature.name()+"["+lengthName+"]/"+feature.rootTypeId()).c_str() : (feature.name()+"/"+feature.rootTypeId()).c_str(),
                    bufferSize 
                );
                branches.push_back(branchData);
            }
            return branches;
        }

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
                tree_->Branch("jetorigin_isLLP_B",&jetorigin_isLLP_B , "jetorigin_isLLP_B/F", bufferSize) ; 
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
            tree_->Branch("global_phi",&global_phi,"global_phi/F",bufferSize);

            globalBranches = makeBranches<0>(tree_,globalFeatures);
            csvBranches = makeBranches<0>(tree_,csvFeatures);

            tree_->Branch("ncpf",&ncpf,"ncpf/I",bufferSize);
            cpfBranches = makeBranches<maxEntries_cpf>(tree_,cpfFeatures,"ncpf");
            
            tree_->Branch("nnpf",&nnpf,"nnpf/I",bufferSize);
            npfBranches = makeBranches<maxEntries_npf>(tree_,npfFeatures,"nnpf");
            
            tree_->Branch("nsv",&nsv,"nsv/I",bufferSize);
            svBranches = makeBranches<maxEntries_sv>(tree_,svFeatures,"nsv");

    	    tree_->Branch("nmuon",&nmuon,"nmuon/I",bufferSize); 
            muonBranches = makeBranches<maxEntries_muon>(tree_,muonFeatures,"nmuon");
            
            tree_->Branch("nelectron",&nelectron,"nelectron/I",bufferSize);
            electronBranches = makeBranches<maxEntries_electron>(tree_,electronFeatures,"nelectron");

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
        
        static constexpr int maxJets = 30; //allows for a maximum of 30 jets per event
        static constexpr int maxEntries_global = maxJets;
        static constexpr int maxEntries_cpf = UnpackedTree::maxEntries_cpf*maxJets;
        static constexpr int maxEntries_npf = UnpackedTree::maxEntries_npf*maxJets;
        static constexpr int maxEntries_sv = UnpackedTree::maxEntries_sv*maxJets;
        static constexpr int maxEntries_muon = UnpackedTree::maxEntries_muon*maxJets;
        static constexpr int maxEntries_electron = UnpackedTree::maxEntries_electron*maxJets;
        
        
        unsigned int nJet;
        float Jet_eta[maxEntries_global];
        float Jet_phi[maxEntries_global];
        float Jet_pt[maxEntries_global];
        unsigned int Jet_jetId[maxEntries_global];
        unsigned int Jet_nConstituents[maxEntries_global];
        
        int Jet_muonIdx1[maxEntries_global];
        int Jet_muonIdx2[maxEntries_global];
        int Jet_electronIdx1[maxEntries_global];
        int Jet_electronIdx2[maxEntries_global];
        
        unsigned int nMuon;        
        float Muon_pt[10];
        unsigned int nElectron;
        float Electron_pt[10];
        
        float Jet_forDA[maxEntries_global];
        int Jet_genJetIdx[maxEntries_global];

        float GenJet_pt[maxEntries_global];
        
        unsigned int njetorigin;
        int jetorigin_jetIdx[maxEntries_global];
        
        int jetorigin_isPU[maxEntries_global];
        int jetorigin_isB[maxEntries_global];
        int jetorigin_isBB[maxEntries_global];
        int jetorigin_isGBB[maxEntries_global];
        int jetorigin_isLeptonic_B[maxEntries_global];
        int jetorigin_isLeptonic_C[maxEntries_global];
        int jetorigin_isC[maxEntries_global];
        int jetorigin_isCC[maxEntries_global];
        int jetorigin_isGCC[maxEntries_global];
        int jetorigin_isS[maxEntries_global];
        int jetorigin_isUD[maxEntries_global];
        int jetorigin_isG[maxEntries_global];

        int jetorigin_isLLP_RAD[maxEntries_global]; 
        int jetorigin_isLLP_MU[maxEntries_global]; 
        int jetorigin_isLLP_E[maxEntries_global]; 
        int jetorigin_isLLP_Q[maxEntries_global];
        int jetorigin_isLLP_QMU[maxEntries_global]; 
        int jetorigin_isLLP_QE[maxEntries_global]; 
        int jetorigin_isLLP_QQ[maxEntries_global]; 
        int jetorigin_isLLP_QQMU[maxEntries_global]; 
        int jetorigin_isLLP_QQE[maxEntries_global]; 
        int jetorigin_isLLP_B[maxEntries_global]; 
        int jetorigin_isLLP_BMU[maxEntries_global];
        int jetorigin_isLLP_BE[maxEntries_global]; 
        int jetorigin_isLLP_BB[maxEntries_global]; 
        int jetorigin_isLLP_BBMU[maxEntries_global]; 
        int jetorigin_isLLP_BBE[maxEntries_global]; 
        int jetorigin_isUndefined[maxEntries_global];
        
        float jetorigin_displacement[maxEntries_global];
        float jetorigin_decay_angle[maxEntries_global];
        float jetorigin_displacement_xy[maxEntries_global];
        float jetorigin_displacement_z[maxEntries_global]; 
        float jetorigin_betagamma[maxEntries_global];
        int   jetorigin_partonFlavor[maxEntries_global];
        int   jetorigin_hadronFlavor[maxEntries_global];
        int   jetorigin_llpId[maxEntries_global];
        float jetorigin_llp_mass[maxEntries_global];
        float jetorigin_llp_pt[maxEntries_global];
        
        unsigned int nlength;
        int length_cpf[maxEntries_global];
        int length_npf[maxEntries_global];
        int length_sv[maxEntries_global];
        int length_muon[maxEntries_global];
        int length_electron[maxEntries_global];
        
        float xsecweight;
        float processId;
        float isData;
        

        unsigned int nglobal;
        int global_jetIdx[maxEntries_global];
        float global_pt[maxEntries_global];
        float global_eta[maxEntries_global];
        float global_phi[maxEntries_global];
        
        std::vector<std::shared_ptr<BranchData>> globalBranches;
        
        unsigned int ncsv;
        std::vector<std::shared_ptr<BranchData>> csvBranches;

        unsigned int ncpf;
        int cpf_jetIdx[maxEntries_cpf];
        std::vector<std::shared_ptr<BranchData>> cpfBranches;

        unsigned int nnpf;
        int npf_jetIdx[maxEntries_npf];
        std::vector<std::shared_ptr<BranchData>> npfBranches;
        
        unsigned int nsv;
        int sv_jetIdx[maxEntries_sv];
        std::vector<std::shared_ptr<BranchData>> svBranches;

        unsigned int nmuon;
        int muon_jetIdx[maxEntries_muon];
        std::vector<std::shared_ptr<BranchData>> muonBranches;

        unsigned int nelectron;
        int electron_jetIdx[maxEntries_electron];
        std::vector<std::shared_ptr<BranchData>> electronBranches;

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
        
        float isB_ANY;
        float isC_ANY;
        float isLLP_ANY;
        
        
        float rand;
        float pt;
        float eta;
        float phi;
        
        float ctau;
        
        Parser parser_;
        SymbolTable symbolTable_;
        std::vector<Expression> selections_;
        std::vector<Expression> setters_;
        
        
        template<size_t N>
        std::vector<std::shared_ptr<BranchData>> branchAddresses(TTree* tree, const std::vector<Feature>& features) const
        {
            std::vector<std::shared_ptr<BranchData>> branches;
            for (size_t ifeature = 0; ifeature < features.size(); ++ifeature)
            {
                auto const& feature = features[ifeature];
                auto branchData = BranchData::branchAddress<N>(
                    feature.type(),
                    tree,
                    feature.name().c_str()
                );
                branches.push_back(branchData);
            }
            return branches;
        }
        
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
            tree_->SetBranchAddress("Jet_phi",&Jet_phi);
            tree_->SetBranchAddress("Jet_pt",&Jet_pt);
            tree_->SetBranchAddress("Jet_jetId",&Jet_jetId);
            tree_->SetBranchAddress("Jet_nConstituents",&Jet_nConstituents);
            tree_->SetBranchAddress("Jet_genJetIdx", &Jet_genJetIdx);
            tree_->SetBranchAddress("GenJet_pt", &GenJet_pt);
            
            tree_->SetBranchAddress("Jet_muonIdx1",&Jet_muonIdx1);
            tree_->SetBranchAddress("Jet_muonIdx2",&Jet_muonIdx2);
            tree_->SetBranchAddress("Jet_electronIdx1",&Jet_electronIdx1);
            tree_->SetBranchAddress("Jet_electronIdx2",&Jet_electronIdx2);
            
            tree_->SetBranchAddress("nMuon",&nMuon);
            tree_->SetBranchAddress("Muon_pt",&Muon_pt);
            tree_->SetBranchAddress("nElectron",&nElectron);
            tree_->SetBranchAddress("Electron_pt",&Electron_pt);
        
            if (addTruth)
            {
                tree_->SetBranchAddress("njetorigin",&njetorigin);
                tree_->SetBranchAddress("jetorigin_jetIdx",&jetorigin_jetIdx);
                
                tree_->SetBranchAddress("jetorigin_displacement",&jetorigin_displacement);
                tree_->SetBranchAddress("jetorigin_decay_angle",&jetorigin_decay_angle);
                tree_->SetBranchAddress("jetorigin_displacement_xy" , &jetorigin_displacement_xy); 
                tree_->SetBranchAddress("jetorigin_displacement_z" , &jetorigin_displacement_z); 
                tree_->SetBranchAddress("jetorigin_betagamma", &jetorigin_betagamma);
                
                tree_->SetBranchAddress("jetorigin_hadronFlavor", &jetorigin_hadronFlavor);
                tree_->SetBranchAddress("jetorigin_partonFlavor", &jetorigin_partonFlavor);
                
                tree_->SetBranchAddress("jetorigin_llpId", &jetorigin_llpId);
                tree_->SetBranchAddress("jetorigin_llp_pt", &jetorigin_llp_pt);
                tree_->SetBranchAddress("jetorigin_llp_mass", &jetorigin_llp_mass);
     
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
                tree_->SetBranchAddress("jetorigin_isLLP_B",&jetorigin_isLLP_B) ; 
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
            tree_->SetBranchAddress("global_jetIdx",&global_jetIdx);
            tree_->SetBranchAddress("global_pt",&global_pt);
            tree_->SetBranchAddress("global_eta",&global_eta);
            tree_->SetBranchAddress("global_phi",&global_phi);

            globalBranches = branchAddresses<maxEntries_global>(tree_,globalFeatures);
            
            tree_->SetBranchAddress("ncsv",&ncsv);
            csvBranches = branchAddresses<maxEntries_global>(tree_,csvFeatures);
            
            tree_->SetBranchAddress("ncpf",&ncpf);
            tree_->SetBranchAddress("cpf_jetIdx",&cpf_jetIdx);
            cpfBranches = branchAddresses<maxEntries_cpf>(tree_,cpfFeatures);
            
            tree_->SetBranchAddress("nnpf",&nnpf);
            tree_->SetBranchAddress("npf_jetIdx",&npf_jetIdx);
            npfBranches = branchAddresses<maxEntries_npf>(tree_,npfFeatures);
            
            tree_->SetBranchAddress("nsv",&nsv);
            tree_->SetBranchAddress("sv_jetIdx",&sv_jetIdx);
            svBranches = branchAddresses<maxEntries_sv>(tree_,svFeatures);
		  
		    tree_->SetBranchAddress("nmuon",&nmuon); 
		    tree_->SetBranchAddress("muon_jetIdx",&muon_jetIdx);
            muonBranches = branchAddresses<maxEntries_muon>(tree_,muonFeatures);
 
            tree_->SetBranchAddress("nelectron",&nelectron); 
            tree_->SetBranchAddress("electron_jetIdx",&electron_jetIdx);
            electronBranches = branchAddresses<maxEntries_electron>(tree_,electronFeatures);
            
            
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
            
            symbolTable_.add_variable("isPU",isPU);
            
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
            
            symbolTable_.add_variable("isB_ANY",isB_ANY);
            symbolTable_.add_variable("isC_ANY",isC_ANY);
            symbolTable_.add_variable("isLLP_ANY",isLLP_ANY);

            symbolTable_.add_variable("rand",rand);
            symbolTable_.add_variable("ctau",ctau);
	    
            symbolTable_.add_variable("pt",pt);
            symbolTable_.add_variable("eta",eta);
            symbolTable_.add_variable("phi",phi);
            
           
            for (auto selectstring: selectors)
            {
                std::cout<<"register selection: "<<selectstring<<std::endl;
                Expression exp;
                exp.register_symbol_table(symbolTable_);
                if (not parser_.compile(selectstring,exp))
                {
                    for (std::size_t i = 0; i < parser_.error_count(); ++i)
                    {
                        auto error = parser_.get_error(i);
                        std::cout<<"Expression compilation error #"<<i<<std::endl;
                        std::cout<<" -> Position: "<<error.token.position;
                        std::cout<<", Type: "<<exprtk::parser_error::to_str(error.mode);
                        std::cout<<", Msg: "<<error.diagnostic<<std::endl<<std::endl;
                    }
                    throw std::runtime_error("Compilation error");
                }
                else
                {
                    selections_.emplace_back(std::move(exp));
                }
            }
            
            for (auto setstring: setters)
            {
                std::cout<<"register setter: "<<setstring<<std::endl;
                Expression exp;
                exp.register_symbol_table(symbolTable_);
                if (not parser_.compile(setstring,exp))
                {
                    for (std::size_t i = 0; i < parser_.error_count(); ++i)
                    {
                        auto error = parser_.get_error(i);
                        std::cout<<"Expression compilation error #"<<i<<std::endl;
                        std::cout<<" -> Position: "<<error.token.position;
                        std::cout<<", Type: "<<exprtk::parser_error::to_str(error.mode);
                        std::cout<<", Msg: "<<error.diagnostic<<std::endl<<std::endl;
                    }
                    throw std::runtime_error("Compilation error");
                }
                else
                {
                    setters_.emplace_back(std::move(exp));
                }
                
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
            return std::min<int>(nJet,maxJets);
        }
        
        bool isSelected(unsigned int jet)
        {
            
            //nJet should be lower than e.g. njetorigin since pT selection on Jet's are applied
            if (jet>=nJet)
            {
                return false;
            }
            
            //reverse search for indices
            int indexGlobal = -1;
            int indexOrigin = -1;
            
            if (nglobal!=nlength)
            {
                std::cout<<"Encountered mismatch between length of candidates and global jets"<<std::endl;
                return false;
            }
            
            for (int ijet = 0; ijet < nglobal; ++ijet)
            {
                if (global_jetIdx[ijet]==jet) indexGlobal = ijet;
                if (addTruth_)
                {
                    if (jetorigin_jetIdx[ijet]==jet) indexOrigin = ijet;
                }
            }
            
            if (indexGlobal<0 or (addTruth_ and indexOrigin<0))
            {
                return false;
            }
            
            //at least 10 GeV uncorrected
            if (global_pt[indexGlobal]<10.)
            {
                return false;
            }
            
            //ignore jet if reco/gen pt largely disagree -> likely random PU match
            //require minimum of genjet pt of 5 GeV
            if (addTruth_ and jetorigin_isPU[indexOrigin]==0 and Jet_genJetIdx[jet]>-1 and Jet_genJetIdx[jet]<maxJets)
            {
                if ((GenJet_pt[Jet_genJetIdx[jet]]<5.) or ((Jet_pt[jet]/GenJet_pt[Jet_genJetIdx[jet]]) < 0.5))
                {
                    //std::cout << "Skipping jet with mismatched genpt: reco pt="<<Jet_pt[jet] << ", genpt="<<GenJet_pt[Jet_genJetIdx[jet]] << std::endl;
                    return false;
                }
            }
            
            if (this->njets()<jet)
            {
                std::cout<<"Not enough jets to unpack"<<std::endl;
                return false;
            }
            
            if (std::fabs(Jet_eta[jet])>2.4)
            {
                return false;
            }
            
            //just a sanity check
            if (std::fabs(Jet_eta[jet]/global_eta[indexGlobal]-1)>0.01 or std::fabs(Jet_phi[jet]/global_phi[indexGlobal]-1)>0.01)
            {
                std::cout<<"Encountered mismatch between standard nanoaod jets and xtag info"<<std::endl;
                return false;
            }
            
            float leptonPtSum = 0.;
            if (Jet_muonIdx1[jet]>=0 and nMuon>Jet_muonIdx1[jet]) leptonPtSum+=Muon_pt[Jet_muonIdx1[jet]];
            if (Jet_muonIdx2[jet]>=0 and nMuon>Jet_muonIdx2[jet]) leptonPtSum+=Muon_pt[Jet_muonIdx2[jet]];
            if (Jet_electronIdx1[jet]>=0 and nElectron>Jet_electronIdx1[jet]) leptonPtSum+=Electron_pt[Jet_electronIdx1[jet]];
            if (Jet_electronIdx2[jet]>=0 and nElectron>Jet_electronIdx2[jet]) leptonPtSum+=Electron_pt[Jet_electronIdx2[jet]];
            
            if ((leptonPtSum/Jet_pt[jet])>0.6) 
            {
                return false;
            }
            
            if (addTruth_)
            {
                if (jetorigin_isUndefined[indexOrigin]>0.5)
                {
                    return false;
                }

                isB = jetorigin_isB[indexOrigin];
                isBB = jetorigin_isBB[indexOrigin];
                isGBB = jetorigin_isGBB[indexOrigin];
                isLeptonic_B = jetorigin_isLeptonic_B[indexOrigin];
                isLeptonic_C = jetorigin_isLeptonic_C[indexOrigin];
                
                isC = jetorigin_isC[indexOrigin];
                isCC = jetorigin_isCC[indexOrigin];
                isGCC = jetorigin_isGCC[indexOrigin];
                
                isS = jetorigin_isS[indexOrigin];
                isUD = jetorigin_isUD[indexOrigin];
                isG = jetorigin_isG[indexOrigin];
 
                isLLP_RAD= jetorigin_isLLP_RAD[indexOrigin];  
                isLLP_MU= jetorigin_isLLP_MU[indexOrigin];  
                isLLP_E= jetorigin_isLLP_E[indexOrigin];  
                isLLP_Q= jetorigin_isLLP_Q[indexOrigin];  
                isLLP_QMU= jetorigin_isLLP_QMU[indexOrigin];  
                isLLP_QE= jetorigin_isLLP_QE[indexOrigin];  
                isLLP_QQ= jetorigin_isLLP_QQ[indexOrigin];  
                isLLP_QQMU= jetorigin_isLLP_QQMU[indexOrigin];  
                isLLP_QQE= jetorigin_isLLP_QQE[indexOrigin];  
                isLLP_B= jetorigin_isLLP_B[indexOrigin];  
                isLLP_BMU= jetorigin_isLLP_BMU[indexOrigin];  
                isLLP_BE= jetorigin_isLLP_BE[indexOrigin];  
                isLLP_BB= jetorigin_isLLP_BB[indexOrigin];  
                isLLP_BBMU= jetorigin_isLLP_BBMU[indexOrigin];  
                isLLP_BBE= jetorigin_isLLP_BBE[indexOrigin];  
                
                isPU = jetorigin_isPU[indexOrigin];
                
                isB_ANY = isB+isBB+isGBB+isLeptonic_B+isLeptonic_C;
                isC_ANY = isC+isCC+isGCC;
                isLLP_ANY = isLLP_RAD+isLLP_MU+isLLP_E+isLLP_Q+isLLP_QMU+isLLP_QE+isLLP_QQ+isLLP_QQMU+isLLP_QQE
                            +isLLP_B+isLLP_BMU+isLLP_BE+isLLP_BB+isLLP_BBMU+isLLP_BBE;
               
                if ((isB_ANY+isC_ANY+
                    +isS+isUD+isG+isPU
                    +isLLP_ANY+jetorigin_isUndefined[indexOrigin])!=1)
                {
                    std::cout<<"Error - label sum is not 1"<<std::endl;
                    std::cout<<"isB: "<<isB<<", isBB: "<<isBB<<", isGBB: "<<isGBB<<", isLeptonic_B: "<<isLeptonic_B<<", isLeptonic_C: "<<isLeptonic_C;
                    std::cout<<", isC: "<<isC<<", isCC: "<<isCC<<", isGCC: "<<isGCC<<", isS: "<<isS<<", isUD: "<<isUD<<", isG: "<<isG<<", isPU: "<<isPU;
                    std::cout<<", isLLP_RAD: "<<isLLP_RAD<<", isLLP_MU: "<<isLLP_MU<<", isLLP_E: "<<isLLP_E<<", isLLP_Q: "<<isLLP_Q;
                    std::cout<<", isLLP_QMU: "<<isLLP_QMU<<", isLLP_QE: "<<isLLP_QE<<", isLLP_QQ: "<<isLLP_QQ<<", isLLP_QQMU: "<<isLLP_QQMU;
                    std::cout<<", isLLP_QQE: "<<isLLP_QQE<<", isLLP_B: "<<isLLP_B<<", isLLP_BMU: "<<isLLP_BMU<<", isLLP_BE: "<<isLLP_BE;
                    std::cout<<", isLLP_BB: "<<isLLP_BB<<", isLLP_BBMU: "<<isLLP_BBMU<<", isLLP_BBE: "<<isLLP_BBE;
                    std::cout<<", isLLP_ANY: "<<isLLP_ANY<<", isUndefined: "<<jetorigin_isUndefined[jet]<<std::endl;
                    return false;
                }
               
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
            ctau = -10;
            pt = global_pt[indexGlobal];
            eta = global_eta[indexGlobal];
            phi = global_phi[indexGlobal];
            
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
            
            int indexOrigin = -1;
            for (int ijet = 0; ijet < nJet; ++ijet)
            {
                if (jetorigin_jetIdx[ijet]==jet) indexOrigin = ijet;
            }
            
            if (indexOrigin==-1) return 0;

            if  (jetorigin_isB[indexOrigin]>0.5) return 0;
            if  (jetorigin_isBB[indexOrigin]>0.5) return 1;
            if  (jetorigin_isGBB[indexOrigin]>0.5) return 2;
            if  (jetorigin_isLeptonic_B[indexOrigin]>0.5) return 3;
            if  (jetorigin_isLeptonic_C[indexOrigin]>0.5) return 4;
            if  (jetorigin_isC[indexOrigin]>0.5) return 5;
            if  (jetorigin_isCC[indexOrigin]>0.5) return 6;
            if  (jetorigin_isGCC[indexOrigin]>0.5) return 7;
            if  (jetorigin_isS[indexOrigin]>0.5) return 8;
            if  (jetorigin_isUD[indexOrigin]>0.5) return 9;
            if  (jetorigin_isG[indexOrigin]>0.5) return 10;
            if  (jetorigin_isPU[indexOrigin]>0.5) return 11;
            if  (jetorigin_isLLP_RAD[indexOrigin]>0.5) return 12;
            if  (jetorigin_isLLP_MU[indexOrigin]>0.5) return 13;
            if  (jetorigin_isLLP_E[indexOrigin]>0.5) return 14;
            if  (jetorigin_isLLP_Q[indexOrigin]>0.5) return 15;
            if  (jetorigin_isLLP_QMU[indexOrigin]>0.5) return 16;
            if  (jetorigin_isLLP_QE[indexOrigin]>0.5) return 17;
            if  (jetorigin_isLLP_QQ[indexOrigin]>0.5) return 18;
            if  (jetorigin_isLLP_QQMU[indexOrigin]>0.5) return 19;
            if  (jetorigin_isLLP_QQE[indexOrigin]>0.5) return 20;
            if  (jetorigin_isLLP_B[indexOrigin]>0.5) return 21;
            if  (jetorigin_isLLP_BMU[indexOrigin]>0.5) return 22;
            if  (jetorigin_isLLP_BE[indexOrigin]>0.5) return 23;
            if  (jetorigin_isLLP_BB[indexOrigin]>0.5) return 24;
            if  (jetorigin_isLLP_BBMU[indexOrigin]>0.5) return 25;
            if  (jetorigin_isLLP_BBE[indexOrigin]>0.5) return 26;

            return -1;
        }
        
        bool unpackJet(
            unsigned int jet,
            UnpackedTree& unpackedTree
        )
        {
            if (this->njets()<jet) return false;
            
            //reverse search for indices
            int indexGlobal = -1;
            int indexOrigin = -1;
            
            for (int ijet = 0; ijet < nglobal; ++ijet)
            {
                if (global_jetIdx[ijet]==jet) indexGlobal = ijet;
                if (addTruth_)
                {
                    if (jetorigin_jetIdx[ijet]==jet) indexOrigin = ijet;
                }
            }
            
            if (indexGlobal<0 or (addTruth_ and indexOrigin<0))
            {
                return false;
            }
            
            if (addTruth_)
            {
                unpackedTree.jetorigin_displacement = jetorigin_displacement[indexOrigin];
                unpackedTree.jetorigin_displacement_xy = jetorigin_displacement_xy[indexOrigin];
                unpackedTree.jetorigin_displacement_z = jetorigin_displacement_z[indexOrigin];
                unpackedTree.jetorigin_ctau = ctau;
                unpackedTree.jetorigin_decay_angle = jetorigin_decay_angle[indexOrigin];
                unpackedTree.jetorigin_betagamma = jetorigin_betagamma[indexOrigin];
                
                unpackedTree.jetorigin_partonFlavor = jetorigin_partonFlavor[indexOrigin];
                unpackedTree.jetorigin_hadronFlavor = jetorigin_hadronFlavor[indexOrigin];
                
                unpackedTree.jetorigin_llpId = jetorigin_llpId[indexOrigin];
                unpackedTree.jetorigin_llp_mass = jetorigin_llp_mass[indexOrigin];
                unpackedTree.jetorigin_llp_pt = jetorigin_llp_pt[indexOrigin];
                
                unpackedTree.jetorigin_isUndefined = jetorigin_isUndefined[indexOrigin];
                unpackedTree.jetorigin_isB = jetorigin_isB[indexOrigin];
                unpackedTree.jetorigin_isBB = jetorigin_isBB[indexOrigin];
                unpackedTree.jetorigin_isGBB = jetorigin_isGBB[indexOrigin];
                unpackedTree.jetorigin_isLeptonic_B = jetorigin_isLeptonic_B[indexOrigin];
                unpackedTree.jetorigin_isLeptonic_C = jetorigin_isLeptonic_C[indexOrigin];
                unpackedTree.jetorigin_isC = jetorigin_isC[indexOrigin];
                unpackedTree.jetorigin_isCC = jetorigin_isCC[indexOrigin];
                unpackedTree.jetorigin_isGCC = jetorigin_isGCC[indexOrigin];
                unpackedTree.jetorigin_isS = jetorigin_isS[indexOrigin];
                unpackedTree.jetorigin_isUD = jetorigin_isUD[indexOrigin];
                unpackedTree.jetorigin_isG = jetorigin_isG[indexOrigin];
                unpackedTree.jetorigin_isPU = jetorigin_isPU[indexOrigin];
                unpackedTree.jetorigin_isLLP_RAD= jetorigin_isLLP_RAD[indexOrigin];
                unpackedTree.jetorigin_isLLP_MU= jetorigin_isLLP_MU[indexOrigin];
                unpackedTree.jetorigin_isLLP_E= jetorigin_isLLP_E[indexOrigin];
                unpackedTree.jetorigin_isLLP_Q= jetorigin_isLLP_Q[indexOrigin];
                unpackedTree.jetorigin_isLLP_QMU= jetorigin_isLLP_QMU[indexOrigin];
                unpackedTree.jetorigin_isLLP_QE= jetorigin_isLLP_QE[indexOrigin];
                unpackedTree.jetorigin_isLLP_QQ= jetorigin_isLLP_QQ[indexOrigin];
                unpackedTree.jetorigin_isLLP_QQMU= jetorigin_isLLP_QQMU[indexOrigin];
                unpackedTree.jetorigin_isLLP_QQE= jetorigin_isLLP_QQE[indexOrigin];
                unpackedTree.jetorigin_isLLP_B= jetorigin_isLLP_B[indexOrigin];
                unpackedTree.jetorigin_isLLP_BMU= jetorigin_isLLP_BMU[indexOrigin];
                unpackedTree.jetorigin_isLLP_BE= jetorigin_isLLP_BE[indexOrigin];
                unpackedTree.jetorigin_isLLP_BB= jetorigin_isLLP_BB[indexOrigin];
                unpackedTree.jetorigin_isLLP_BBMU= jetorigin_isLLP_BBMU[indexOrigin];
                unpackedTree.jetorigin_isLLP_BBE= jetorigin_isLLP_BBE[indexOrigin];		
            }
            else
            {
                unpackedTree.isData = isData;
                unpackedTree.xsecweight = xsecweight;
                unpackedTree.processId = processId;
            }
            
            
            unpackedTree.global_pt = global_pt[indexGlobal];
            unpackedTree.global_eta = global_eta[indexGlobal];
            unpackedTree.global_phi = global_phi[indexGlobal];

            if (unpackedTree.globalBranches.size()!=globalBranches.size()) throw std::runtime_error("Global branches have different size!");
            for (size_t ifeature = 0; ifeature < globalBranches.size(); ++ifeature)
            {
                unpackedTree.globalBranches[ifeature]->setFloat(0,globalBranches[ifeature]->getFloat(indexGlobal));
            }
            
            if (unpackedTree.csvBranches.size()!=csvBranches.size()) throw std::runtime_error("CSV branches have different size!");
            for (size_t ifeature = 0; ifeature < csvBranches.size(); ++ifeature)
            {
                unpackedTree.csvBranches[ifeature]->setFloat(0,csvBranches[ifeature]->getFloat(indexGlobal));
            }

            
            int cpf_offset = 0;
            for (size_t i = 0; i < indexGlobal; ++i)
            {
                cpf_offset += length_cpf[i];                
            }            
            if (length_cpf[indexGlobal]>0 and jet!=cpf_jetIdx[cpf_offset])
            {
                throw std::runtime_error("CPF jet index different than global one");
            }
            
            int ncpf = std::min<int>(UnpackedTree::maxEntries_cpf,length_cpf[indexGlobal]);
            unpackedTree.ncpf = ncpf;
            
            if (unpackedTree.cpfBranches.size()!=cpfBranches.size()) throw std::runtime_error("CPF branches have different size!");
            
            for (size_t ifeature = 0; ifeature < cpfBranches.size(); ++ifeature)
            {
                for (int i = 0; i < ncpf; ++i)
                {
                    unpackedTree.cpfBranches[ifeature]->setFloat(i,cpfBranches[ifeature]->getFloat(cpf_offset+i));
                }
            }
            
            
            int npf_offset = 0;
            for (size_t i = 0; i < indexGlobal; ++i)
            {
                npf_offset += length_npf[i];
            }
            if (length_npf[indexGlobal]>0 and jet!=npf_jetIdx[npf_offset])
            {
                throw std::runtime_error("NPF jet index different than global one");
            }

            int nnpf = std::min<int>(UnpackedTree::maxEntries_npf,length_npf[indexGlobal]);
            unpackedTree.nnpf = nnpf;
            
            if (unpackedTree.npfBranches.size()!=npfBranches.size()) throw std::runtime_error("NPF branches have different size!");
            
            for (size_t ifeature = 0; ifeature < npfBranches.size(); ++ifeature)
            {
                for (int i = 0; i < nnpf; ++i)
                {
                    unpackedTree.npfBranches[ifeature]->setFloat(i,npfBranches[ifeature]->getFloat(npf_offset+i));
                }   
            }


            int sv_offset = 0;
            for (size_t i = 0; i < indexGlobal; ++i)
            {
                sv_offset += length_sv[i];
            }
            if (length_sv[indexGlobal]>0 and jet!=sv_jetIdx[sv_offset])
            {
                throw std::runtime_error("SV jet index different than global one");
            }
            
            int nsv = std::min<int>(UnpackedTree::maxEntries_sv,length_sv[indexGlobal]);
            unpackedTree.nsv = nsv;
            
            if (unpackedTree.svBranches.size()!=svBranches.size()) throw std::runtime_error("SV branches have different size!");
            
            for (size_t ifeature = 0; ifeature < svBranches.size(); ++ifeature)
            {
                for (int i = 0; i < nsv; ++i)
                {
                    unpackedTree.svBranches[ifeature]->setFloat(i,svBranches[ifeature]->getFloat(sv_offset+i));
                }
            }
            
            
            int muon_offset = 0;
            for (size_t i = 0; i < indexGlobal; ++i)
            {
                muon_offset += length_muon[i];
            }
            if (length_muon[indexGlobal]>0 and jet!=muon_jetIdx[muon_offset])
            {
                throw std::runtime_error("Muon jet index different than global one");
            }
            
            int nmuon = std::min<int>(UnpackedTree::maxEntries_muon,length_muon[indexGlobal]);
            unpackedTree.nmuon = nmuon;
            
            if (unpackedTree.muonBranches.size()!=muonBranches.size()) throw std::runtime_error("Muon branches have different size!");
            for (size_t ifeature = 0; ifeature < muonBranches.size(); ++ifeature)
            {
                for (int i = 0; i < nmuon; ++i)
                {
                    unpackedTree.muonBranches[ifeature]->setFloat(i,muonBranches[ifeature]->getFloat(muon_offset+i));
	            }
            }


            int electron_offset = 0;
            for (size_t i = 0; i < indexGlobal; ++i)
            {
                electron_offset += length_electron[i];
            }
            if (length_electron[indexGlobal]>0 and jet!=electron_jetIdx[electron_offset])
            {
                throw std::runtime_error("Electron jet index different than global one");
            }
            
            int nelectron = std::min<int>(UnpackedTree::maxEntries_electron,length_electron[indexGlobal]);
            unpackedTree.nelectron = nelectron;

            if (unpackedTree.electronBranches.size()!=electronBranches.size()) throw std::runtime_error("Electron branches have different size!");
            for (size_t ifeature = 0; ifeature < electronBranches.size(); ++ifeature)
            {
                for (int i = 0; i < nelectron; ++i)
                {
                    unpackedTree.electronBranches[ifeature]->setFloat(i,electronBranches[ifeature]->getFloat(electron_offset+i));
                }
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
    std::vector<std::vector<int>> eventsPerClassPerFileTrain(27,std::vector<int>(nOutputs,0));
    
    std::vector<std::unique_ptr<UnpackedTree>> unpackedTreesTest;
    std::vector<std::vector<int>> eventsPerClassPerFileTest(27,std::vector<int>(nOutputs,0));

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
    std::vector<int> writtenJets(entries.size(),0);
    std::vector<int> skippedJets(entries.size(),0);
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
        
        //take only the 6 hardest jets
        for (size_t j = 0; j < std::min<size_t>(6,trees[ifile]->njets()); ++j)
        {
            if (trees[ifile]->isSelected(j))
            {
                int jet_class = trees[ifile]->getJetClass(j);
                long hashTest = calcHash(97*trees[ifile]->entry()+79*j)%100;
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
                        
                        if (trees[ifile]->unpackJet(j,*unpackedTreesTest[ofile]))
                        {
                            eventsPerClassPerFileTest[jet_class][ofile]+=1;
                            writtenJets[ifile]+=1;
                        }
                        else
                        {
                            skippedJets[ifile]+=1;
                        }
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
                        if (trees[ifile]->unpackJet(j,*unpackedTreesTrain[ofile]))
                        {
                            eventsPerClassPerFileTrain[jet_class][ofile]+=1;
                            writtenJets[ifile]+=1;
                        }
                        else
                        {
                            skippedJets[ifile]+=1;
                        }
                    }
                }
            }
            else
            {
                skippedJets[ifile]+=1;
            }
            
        }
    }
    
    for (size_t i = 0; i < entries.size(); ++i)
    {
        std::cout<<"infile "<<inputs[i]<<":"<<std::endl;
        std::cout<<"\tevents: found = "<<entries[i]<<", read = "<<readEvents[i]<<"/"<<int(1.*entries[i]/nSplit)<<std::endl;
        std::cout<<"\tjets: written = "<<writtenJets[i]<<", skipped = "<<skippedJets[i]<<std::endl;
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
