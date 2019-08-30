import tensorflow as tf

featureDict = {
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
    "truth": {
        "branches":[
            'jetorigin_isB||jetorigin_isBB||jetorigin_isGBB||jetorigin_isLeptonic_B||jetorigin_isLeptonic_C',         
            'jetorigin_isC||jetorigin_isCC||jetorigin_isGCC',
            'jetorigin_isUD||jetorigin_isS',
            'jetorigin_isG',
            'jetorigin_fromLLP',
        ],
    },
    
    "gen": {
        "branches":[
            #"jetorigin_ctau",
            "jetorigin_displacement"
        ]
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
            'csv_jetNTracksEtaRel'
            #'legacyTag_median_dxy',
            #'legacyTag_median_trackSip2dSig',
            #'legacyTag_alpha'
        ],
        "preprocessing":{
            #'global_pt':lambda x: tf.log10(tf.nn.relu(x)+1e-3),
            'global_eta':lambda x: tf.abs(x),
            'csv_jetNSelectedTracks': lambda x: x*0.02,
            'csv_trackSip2dValAboveCharm': lambda x: tf.sign(x)*(4+tf.log(tf.abs(x)+1e-6)),
            'csv_trackSip2dSigAboveCharm': lambda x: tf.sign(x)*(1+tf.log(tf.abs(x)+0.1)),
            'csv_trackSip3dValAboveCharm': lambda x: tf.sign(x)*(4+tf.log(tf.abs(x)+1e-6)),
            'csv_trackSip3dSigAboveCharm': lambda x: tf.sign(x)*(tf.log(tf.abs(x)+1)),
            'csv_trackSumJetDeltaR': lambda x: tf.sign(x)*(3+tf.log(tf.abs(x)+1e-3)),
            'ncpf': lambda x: x*0.02,
            'nnpf': lambda x: x*0.02,
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
            'cpf_drminsv',
            'cpf_vertex_association',
            'cpf_fromPV',
            'cpf_puppi_weight',
            'cpf_track_chi2',
            'cpf_track_ndof',
            'cpf_track_quality'
        ],
        "preprocessing":{
            'cpf_trackEtaRel':lambda x: tf.log(1+tf.abs(x)),
            'cpf_trackPtRel':lambda x: tf.log(1e-1+tf.nn.relu(1-x)),
            'cpf_trackPPar': lambda x: tf.log(1e-2+tf.nn.relu(x)),
            'cpf_trackPParRatio': lambda x: tf.log(1e-4+tf.nn.relu(1-x))*0.1,
            'cpf_track_chi2':lambda x: tf.log(1e-2+tf.nn.relu(x)),
            'cpf_trackDeltaR':lambda x: 0.1/(0.1+tf.nn.relu(x)),
            'cpf_trackJetDistVal': lambda x: tf.log(tf.nn.relu(-x)+1e-5),
            'cpf_trackSip2dVal':lambda x: tf.log(tf.nn.relu(x)+1e-3),
            'cpf_trackSip2dSig':lambda x: tf.log(tf.nn.relu(x)+1e-3),
            'cpf_trackSip3dVal':lambda x: tf.log(tf.nn.relu(x)+1e-3),
            'cpf_trackSip3dSig':lambda x: tf.log(tf.nn.relu(x)+1e-3),
            'cpf_track_ndof':lambda x: x*0.05,
            'cpf_ptrel':lambda x: tf.log(1e-2+tf.nn.relu(1-x)),
            
        },
        "max":25
    },
    
    "npf": {
        "branches": [
            'npf_ptrel',
            'npf_deltaR',
            'npf_isGamma',
            'npf_hcal_fraction',
            'npf_drminsv',
            'npf_puppi_weight'
        ],
        "preprocessing":{
            'npf_ptrel':lambda x: tf.log(1e-6+tf.nn.relu(x)),
            'npf_deltaR':lambda x: tf.log(1e-6+tf.nn.relu(x)),
        },
        "max":25
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
            'sv_enratio'
        ],
        "preprocessing":{
            'sv_pt':lambda x: tf.log(tf.nn.relu(x)+1e-3),
            'sv_mass':lambda x: tf.log(tf.nn.relu(x)+1e-3),
            'sv_chi2':lambda x: tf.log(tf.nn.relu(x)+1e-6),
        },
        "max":4
    },
}


