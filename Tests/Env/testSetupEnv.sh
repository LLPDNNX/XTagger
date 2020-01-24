#!/bin/bash

function execute() 
{
    source Env/setupEnv.sh "testenv_cpu"
    source Env/env_cpu.sh || return 1
    python Tests/Env/testKeras.py || return 1
    source deactivate tf_cpu || return 1
}

execute
