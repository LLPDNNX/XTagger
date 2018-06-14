#!/bin/bash

function execute() 
{
    source Env/env_cpu.sh
    cd Unpacker
    cmake . || return 1
    make || return 1
    ./unpackNanoXFast unpacked 5 20 2 1 test-files.txt
}

execute
