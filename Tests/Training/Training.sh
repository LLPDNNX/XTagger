#!/bin/bash

function execute() 
{
    PROJECTDIR=$PWD
    source Env/env_cpu.sh || return 1
    git clone https://github.com/LLPDNNX/test-files || return 1
    export PYTHONPATH=$PYTHONPATH:$PROJECTDIR/Ops/release/lib/python2.7/site-packages
    
    python Training/training.py --name parametric -b 10 -c -p --train test-files/nanox_unpacked/train.txt --test test-files/nanox_unpacked/test.txt -e 1 || return 1 
    python Training/training.py --name ctaux -b 10 -c --train test-files/nanox_unpacked/train.txt --test test-files/nanox_unpacked/test.txt -e 1 || return 1 
    python Training/training.py --name nobalance -b 10 --train test-files/nanox_unpacked/train.txt --test test-files/nanox_unpacked/test.txt -e 1 || return 1 
    python Training/convert_to_const_graph.py output/ctaux/epoch_0/model_epoch.hdf5 || return 1
    python Training/convert_to_const_graph.py -p output/parametric/epoch_0/model_epoch.hdf5 || return 1
}

execute
