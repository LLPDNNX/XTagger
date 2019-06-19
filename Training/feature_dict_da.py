featureDictDA = {
    "isData": {
        "branches":[
            'isData'
        ]
    },
    
    "xsecweight": {
        "branches":[
            'xsecweight'
        ]
    },

     "sv" : {
        "branches":[
            'sv_pt',
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
            'sv_enratio',
            
        ],
        "max":4
    },
    
    "globalvars": {
        "branches": [
            'global_pt',
            'global_eta',
            'ncpf',
            'nnpf',
            'nsv',
            'csv_trackSumJetEtRatio', 
            'csv_trackSumJetDeltaR', 
            'csv_vertexCategory', 
            'csv_trackSip2dValAboveCharm', 
            'csv_trackSip2dSigAboveCharm', 
            'csv_trackSip3dValAboveCharm', 
            'csv_trackSip3dSigAboveCharm', 
            'csv_jetNSelectedTracks', 
            'csv_jetNTracksEtaRel',
            'legacyTag_median_dxy',
            'legacyTag_median_trackSip2dSig',
            'legacyTag_alpha'
        ],

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
            'cpf_drminsv',
            'cpf_vertex_association',
            'cpf_fromPV',
            'cpf_puppi_weight',
            'cpf_track_chi2',
            'cpf_track_ndof',
            'cpf_track_quality',
            #added to test
            #'cpf_jetmassdroprel',
            #'cpf_relIso01',
            #'cpf_isLepton',
            #'cpf_lostInnerHits'

        ],
        "max":25
    },
    
    "npf": {
        "branches": [
            'npf_ptrel',
            'npf_deltaR',
            'npf_isGamma',
            'npf_hcal_fraction',
            'npf_drminsv',
            'npf_puppi_weight',
            # added
            #'npf_jetmassdroprel',
            #'npf_relIso01'

        ],
        "max":25
    }
}

