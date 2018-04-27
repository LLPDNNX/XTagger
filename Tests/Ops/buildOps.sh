#!/bin/bash

function execute() 
{
    source Env/env_cpu.sh || return 1
    source activate tf_cpu || return 1
    mkdir Ops/build || return 1
    cd Ops/build || return 1
    cmake .. || return 1
    make || return 1
}

execute
