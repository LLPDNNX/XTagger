import tensorflow as tf

'''
        'names':[
            'B','C','UDS','G','PU','LLP_QMU','LLP_Q',
        ],
        'weights':[
            'jetorigin_isB||jetorigin_isBB||jetorigin_isGBB||jetorigin_isLeptonic_B||jetorigin_isLeptonic_C',         
            'jetorigin_isC||jetorigin_isCC||jetorigin_isGCC',
            'jetorigin_isUD||jetorigin_isS',
            'jetorigin_isG',
            'jetorigin_isPU*(global_pt<50.)',
            'jetorigin_isLLP_QMU||jetorigin_isLLP_QQMU',
            '(jetorigin_isLLP_QQ||jetorigin_isLLP_Q)*(global_pt<50.)',
        ],
        "branches":[
            'jetorigin_isB||jetorigin_isBB||jetorigin_isGBB||jetorigin_isLeptonic_B||jetorigin_isLeptonic_C',         
            'jetorigin_isC||jetorigin_isCC||jetorigin_isGCC',
            'jetorigin_isUD||jetorigin_isS',
            'jetorigin_isG',
            'jetorigin_isPU',
            'jetorigin_isLLP_QMU||jetorigin_isLLP_QQMU',
            'jetorigin_isLLP_QQ+jetorigin_isLLP_Q'
        ],
    },
'''

featureDict = {
    "truth": {
        'names':[
            'B','C','UDS','G','PU','LLP',
        ],
        'weights':[
            'jetorigin_isB||jetorigin_isBB||jetorigin_isGBB||jetorigin_isLeptonic_B||jetorigin_isLeptonic_C',         
            'jetorigin_isC||jetorigin_isCC||jetorigin_isGCC',
            'jetorigin_isUD||jetorigin_isS',
            'jetorigin_isG',
            'jetorigin_isPU*(global_pt<50.)',
            'jetorigin_isLLP_QMU||jetorigin_isLLP_QQMU||jetorigin_isLLP_QQ||jetorigin_isLLP_Q',
        ],
        "branches":[
            'jetorigin_isB||jetorigin_isBB||jetorigin_isGBB||jetorigin_isLeptonic_B||jetorigin_isLeptonic_C',         
            'jetorigin_isC||jetorigin_isCC||jetorigin_isGCC',
            'jetorigin_isUD||jetorigin_isS',
            'jetorigin_isG',
            'jetorigin_isPU',
            'jetorigin_isLLP_QMU||jetorigin_isLLP_QQMU||jetorigin_isLLP_QQ||jetorigin_isLLP_Q'
        ],
    },
    
    "gen": {
        "branches":[
            "jetorigin_displacement_xy"
        ]
    },
    
    "globalvars": {
        "branches": [
            'global_pt',
            'global_eta',
            'global_mass',
            'global_area',
            'global_n60',
            'global_n90',
            'global_chargedEmEnergyFraction',
            'global_chargedHadronEnergyFraction',
            'global_chargedMuEnergyFraction',
            'global_electronEnergyFraction',
            
            'global_tau1',
            'global_tau2',
            'global_tau3',
            
            'global_relMassDropMassAK',
            'global_relMassDropMassCA',
            'global_relSoftDropMassAK',
            'global_relSoftDropMassCA',
            
            'global_thrust',
            'global_sphericity',
            'global_circularity',
            'global_isotropy',
            'global_eventShapeC',
            'global_eventShapeD',
            
            'csv_trackSumJetEtRatio', 
            'csv_trackSumJetDeltaR', 
            'csv_vertexCategory', 
            'csv_trackSip2dValAboveCharm', 
            'csv_trackSip2dSigAboveCharm', 
            'csv_trackSip3dValAboveCharm', 
            'csv_trackSip3dSigAboveCharm', 
            'csv_jetNSelectedTracks', 
            'csv_jetNTracksEtaRel'
        ],
        "preprocessing":{
            'global_pt':lambda x: tf.log(tf.nn.relu(x)+1e-3),
            'global_mass':lambda x: tf.log(tf.nn.relu(x)+1e-3),
            
            'csv_trackSip2dValAboveCharm':lambda x: tf.sign(x)*(tf.log(tf.abs(x)+1e-3)+5), 
            'csv_trackSip2dSigAboveCharm':lambda x: tf.log(tf.abs(x)+1e-3),
            'csv_trackSip3dValAboveCharm':lambda x: tf.sign(x)*(tf.log(tf.abs(x)+1e-3)+5), 
            'csv_trackSip3dSigAboveCharm':lambda x: tf.log(tf.abs(x)+1e-3),
        },
    },

    "cpf": {
        "branches": [
            'cpf_trackEtaRel',
            'cpf_trackPtRel',
            'cpf_trackPPar',
            'cpf_trackDeltaR',
            'cpf_trackPParRatio',
            'cpf_trackSip2dVal',
            'cpf_trackSip2dSig',
            'cpf_trackSip3dVal',
            'cpf_trackSip3dSig',
            'cpf_trackJetDistVal',
            'cpf_ptrel', 
            'cpf_deta', 
            'cpf_dphi', 
            'cpf_drminsv',
            'cpf_vertex_association',
            'cpf_fromPV',
            'cpf_track_chi2',
            'cpf_track_ndof',
            'cpf_relmassdrop',
            'cpf_track_quality',
            
            'cpf_matchedMuon',
            'cpf_matchedElectron',
            'cpf_matchedSV',
        ],
        "preprocessing":{
            'cpf_ptrel':lambda x: tf.log(1e-6+tf.nn.relu(x)),
            'cpf_deta':lambda x: tf.abs(x),
            'cpf_dphi':lambda x: tf.abs(x),
        
            'cpf_trackEtaRel':lambda x: tf.log(1+tf.abs(x)),
            'cpf_trackPtRel':lambda x: tf.log(1e-1+tf.nn.relu(1-x)),
            'cpf_trackPPar': lambda x: tf.log(1e-2+tf.nn.relu(x)),
            'cpf_trackPParRatio': lambda x: tf.log(1e-4+tf.nn.relu(1-x))*0.1,
            'cpf_track_chi2':lambda x: tf.log(tf.nn.relu(x)+1e-6),
            'cpf_trackDeltaR':lambda x: 0.1/(0.1+tf.nn.relu(x)),
            'cpf_trackJetDistVal': lambda x: tf.log(tf.nn.relu(-x)+1e-5),
            'cpf_trackSip2dVal':lambda x: tf.sign(x)*(tf.log(tf.abs(x)+1e-3)+5),
            'cpf_trackSip2dSig':lambda x: tf.log(tf.abs(x)+1e-3),
            'cpf_trackSip3dVal':lambda x: tf.sign(x)*(tf.log(tf.abs(x)+1e-3)+5),
            'cpf_trackSip3dSig':lambda x: tf.log(tf.abs(x)+1e-3),
            'cpf_track_ndof':lambda x: x*0.05,
            
            
            
        },
        "max":25
    },
    
    "npf": {
        "branches": [
            'npf_ptrel',
            'npf_deta',
            'npf_dphi',
            'npf_deltaR',
            'npf_isGamma',
            'npf_hcal_fraction',
            'npf_drminsv',
            'npf_puppi_weight',
            'npf_relmassdrop',
        ],
        "preprocessing":{
            'npf_ptrel':lambda x: tf.log(1e-6+tf.nn.relu(x)),
            'npf_deta':lambda x: tf.abs(x),
            'npf_dphi':lambda x: tf.abs(x),
            
            'npf_deltaR':lambda x: tf.log(1e-6+tf.nn.relu(x)),
            
        },
        "max":25
    },

    "sv" : {
        "branches":[
            'sv_ptrel',
            'sv_deta',
            'sv_dphi',
            'sv_deltaR',
            'sv_mass',
            'sv_ntracks',
            'sv_chi2',
            'sv_ndof',
            'sv_dxy',
            'sv_dxysig',
            'sv_d3d',
            'sv_d3dsig',
            'sv_costhetasvpv',
            'sv_enratio'
        ],
        "preprocessing":{
            'sv_ptrel':lambda x: tf.log(1e-6+tf.nn.relu(x)),
            'sv_deta':lambda x: tf.abs(x),
            'sv_dphi':lambda x: tf.abs(x),
            'sv_chi2':lambda x: tf.log(tf.nn.relu(x)+1e-6),
            'sv_deltaR':lambda x: tf.log(1e-6+tf.nn.relu(x)),
            
            'sv_dxy':lambda x: tf.log(1e-6+tf.nn.relu(x)),
            'sv_dxysig':lambda x: tf.log(1e-6+tf.nn.relu(x)),
            'sv_d3d':lambda x: tf.log(1e-6+tf.nn.relu(x)),
            'sv_d3dsig':lambda x: tf.log(1e-6+tf.nn.relu(x)),
        },
        "max":4
    },
    
    "muon" : {
        "branches":[
            'muon_ptrel',
            'muon_jetDeltaR',
            'muon_deta',
            'muon_dphi',
            'muon_numberOfMatchedStations',
            'muon_numberOfValidPixelHits',
            'muon_numberOfpixelLayersWithMeasurement',
            'muon_2dIp',
            'muon_2dIpSig',
            'muon_3dIp',
            'muon_3dIpSig',
            'muon_chi2',
            'muon_ndof',
            'muon_caloIso',
            'muon_ecalIso',
            'muon_hcalIso',
            'muon_sumPfChHadronPt',
            'muon_sumPfNeuHadronEt',
            'muon_Pfpileup',
            'muon_sumPfPhotonEt',
            
        ],
        "preprocessing":{
            'muon_ptrel':lambda x: tf.log(1e-6+tf.nn.relu(x)),
            'muon_deta': lambda x: tf.abs(x),
            'muon_dphi': lambda x: tf.abs(x),
            'muon_jetDeltaR':lambda x: tf.log(1e-6+tf.nn.relu(x)),
            
            'muon_2dIp':lambda x: tf.sign(x)*(tf.log(tf.abs(x)+1e-3)+5),
            'muon_2dIpSig':lambda x: tf.log(tf.abs(x)+1e-3),
            'muon_3dIp':lambda x: tf.sign(x)*(tf.log(tf.abs(x)+1e-3)+5),
            'muon_3dIpSig':lambda x: tf.log(tf.abs(x)+1e-3),
            
            'muon_chi2':lambda x: tf.log(tf.nn.relu(x)+1e-6),
            'muon_caloIso':lambda x: tf.log(tf.nn.relu(x)+1e-6),
            'muon_ecalIso':lambda x: tf.log(tf.nn.relu(x)+1e-6),
            'muon_hcalIso':lambda x: tf.log(tf.nn.relu(x)+1e-6),
            'muon_sumPfChHadronPt':lambda x: tf.log(tf.nn.relu(x)+1e-6),
            'muon_sumPfNeuHadronEt':lambda x: tf.log(tf.nn.relu(x)+1e-6),
            'muon_Pfpileup':lambda x: tf.log(tf.nn.relu(x)+1e-6),
            'muon_sumPfPhotonEt':lambda x: tf.log(tf.nn.relu(x)+1e-6),
        },
        "max":2
    },
    
    "electron" : {
        "branches":[
            'electron_ptrel',
            
            'electron_jetDeltaR',
            'electron_deta',
            'electron_dphi',
            'electron_isPassConversionVeto',
            #'electron_convDist',
            'electron_hadronicOverEm',
            
            'electron_nbOfMissingHits',
            
            'electron_2dIP',
            'electron_2dIPSig',
            'electron_3dIP',
            'electron_3dIPSig',
            
            'electron_numberOfBrems',
            #'electron_trackFbrem',
            #'electron_fbrem',
            
            'electron_eSeedClusterOverP',
            'electron_eSeedClusterOverPout',
            
            'electron_SC_energy',
            'electron_SC_eSuperClusterOverP',
            'electron_eTopOvere5x5',
            #'electron_scE1x5Overe5x5',
            #'electron_scE2x5MaxOvere5x5',
            #'electron_scE5x5Rel',
            #'electron_scSigmaEtaEta',
            #'electron_scSigmaIEtaIEta',
            #'electron_superClusterFbrem',
            
            'electron_neutralHadronIso',
            'electron_particleIso',
            'electron_photonIso',
            'electron_puChargedHadronIso',
            'electron_trackIso',
            'electron_ecalPFClusterIso',
            'electron_hcalPFClusterIso',
            
            'electron_hcalDepth1OverEcal',
            'electron_hcalDepth2OverEcal',
            
        ],
        "preprocessing":{
            'electron_ptrel':lambda x: tf.log(1e-6+tf.nn.relu(x)),
            'electron_deta': lambda x: tf.abs(x),
            'electron_dphi': lambda x: tf.abs(x),
            'electron_jetDeltaR':lambda x: tf.log(1e-6+tf.nn.relu(x)),
            
            'electron_hadronicOverEm': lambda x: tf.log(1./(1e-2+tf.nn.relu(x))),
            'electron_eSeedClusterOverP':lambda x: tf.log(1e-5+tf.nn.relu(1-x)),
            'electron_eSeedClusterOverPout':lambda x: tf.log(1e-5+tf.nn.relu(1-x)),
            
            'electron_SC_eSuperClusterOverP':lambda x: tf.log(1e-5+tf.nn.relu(x)),
            
            'electron_2dIP':lambda x: tf.sign(x)*(tf.log(tf.abs(x)+1e-3)+5),
            'electron_2dIPSig':lambda x: tf.log(tf.abs(x)+1e-3),
            'electron_3dIP':lambda x: tf.sign(x)*(tf.log(tf.abs(x)+1e-3)+5),
            'electron_3dIPSig':lambda x: tf.log(tf.abs(x)+1e-3),
            
            #'electron_fbrem': lambda x: tf.nn.relu(x+1),
            
            
            'electron_neutralHadronIso':lambda x: tf.log(tf.nn.relu(x)+1e-6),
            'electron_photonIso':lambda x: tf.log(tf.nn.relu(x)+1e-6),
            'electron_puChargedHadronIso':lambda x: tf.log(tf.nn.relu(x)+1e-6),
            'electron_trackIso':lambda x: tf.log(tf.nn.relu(x)+1e-6),
            'electron_ecalPFClusterIso':lambda x: tf.log(tf.nn.relu(x)+1e-6),
            'electron_hcalPFClusterIso':lambda x: tf.log(tf.nn.relu(x)+1e-6),
            
            'electron_hcalDepth1OverEcal':lambda x: tf.log(tf.nn.relu(x)+1e-6),
            'electron_hcalDepth2OverEcal':lambda x: tf.log(tf.nn.relu(x)+1e-6),
        },
        "max":2
    },
    
}

            

