#!/bin/bash

function execute() 
{
    PROJECTDIR=$PWD
    source Env/env.sh || return 1
    mkdir Ops/build || return 1
    cd Ops/build || return 1
    cmake -DCMAKE_INSTALL_PREFIX=$PROJECTDIR/Ops/release .. || return 1
    make || return 1
    make install || return 1
    export PYTHONPATH=$PROJECTDIR/Ops/release/lib/python2.7/site-packages
    python -c "import xtagger" || return 1
}

execute
