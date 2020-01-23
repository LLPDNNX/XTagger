#!/bin/bash

function execute() 
{
    source Env/setupEnv.sh "testenv_cpu"
    source Env/env.sh || return 1
    python Tests/Env/testKeras.py || return 1
    source deactivate tf || return 1
}

execute
