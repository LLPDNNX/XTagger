#!/bin/bash

function execute() 
{
    source Training/Env/setupEnvCPU.sh "testenv_cpu"
    source Training/Env/env_cpu.sh || return 1
    source activate tf_cpu || return 1
    python Tests/Env/testKeras.py || return 1
    source deactivate tf_cpu || return 1
}

execute
