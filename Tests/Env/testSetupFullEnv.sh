#!/bin/bash

function execute() 
{
    source Training/Env/setupEnvFull.sh "testenv_full"
    source Training/Env/env_cpu.sh || return 1
    source activate tf_cpu || return 1
    python Tests/Env/testKeras.py || return 1
    source deactivate tf_cpu || return 1
    
    source activate tf_gpu || return 1
    source deactivate tf_gpu || return 1
}

execute
