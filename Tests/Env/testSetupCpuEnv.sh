#!/bin/bash

function execute() 
{
    source Env/setupEnvCPU.sh "testenv_cpu"
    source Env/env_cpu.sh || return 1
    python Tests/Env/testKeras.py || return 1
    conda deactivate tf_cpu || return 1
}

execute
