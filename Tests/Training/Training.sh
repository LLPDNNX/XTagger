#!/bin/bash

function execute() 
{
    PROJECTDIR=$PWD
    source Env/env_cpu.sh || return 1
    git clone https://github.com/LLPDNNX/test-files || return 1
    export PYTHONPATH=$PYTHONPATH:$PROJECTDIR/Ops/release/lib/python2.7/site-packages
    export OMP_NUM_THREADS=1
    KERAS_BACKEND=tensorflow python -c "from keras import backend"
    python Training/training.py --noda --name parametric -b 50 -c -p --train test-files/nanox_unpacked/train.txt --test test-files/nanox_unpacked/test.txt -e 2 || return 1 
    python Training/training.py --noda --name ctaux -b 50 -c --train test-files/nanox_unpacked/train.txt --test test-files/nanox_unpacked/test.txt -e 2 || return 1 
    python Training/training.py --noda --name nobalance -b 50 --train test-files/nanox_unpacked/train.txt --test test-files/nanox_unpacked/test.txt -e 2 || return 1 
    
    python Training/convert_to_const_graph.py output/ctaux/epoch_1/model_class.hdf5 || return 1
    python Training/convert_to_const_graph.py -p output/parametric/epoch_1/model_class.hdf5 || return 1
    
    #python Training/training_da.py --name da -b 20 --train test-files/nanox_unpacked/train.txt --test test-files/nanox_unpacked/test.txt --trainDA test-files/nanox_unpacked/da_train.txt --testDA test-files/nanox_unpacked/da_test.txt -e 2 || return 1 
    #python Training/training_da.py --name da_bagging -b 20 --bagging 0.5 --train test-files/nanox_unpacked/train.txt --test test-files/nanox_unpacked/test.txt --trainDA test-files/nanox_unpacked/da_train.txt --testDA test-files/nanox_unpacked/da_test.txt -e 2 || return 1 
    #python Training/training_da.py --name da_wasser --wasserstein -b 20 --train test-files/nanox_unpacked/train.txt --test test-files/nanox_unpacked/test.txt --trainDA test-files/nanox_unpacked/da_train.txt --testDA test-files/nanox_unpacked/da_test.txt -e 2 || return 1 
}

execute
