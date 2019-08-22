#!/bin/bash

function execute() 
{
    source Env/setupEnvFull.sh "testenv_full"
    source Env/env_cpu.sh || return 1
    python Tests/Env/testKeras.py || return 1
    source deactivate tf_cpu || return 1
    
    source Env/env_gpu.sh || return 1
    source deactivate tf_gpu || return 1
}

execute
