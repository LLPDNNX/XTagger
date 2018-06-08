#!/bin/bash

function execute() 
{
    source Env/env_cpu.sh
    cd Unpacker
    cmake . || return 1
    make || return 1
    ./Unpacker/unpackNanoXFast unpacked 5 20 20 1 Unpacker/test-files.txt
}

execute
