#!/bin/bash

function execute() 
{
    source Env/env_cpu.sh
    cd Unpacker
    cmake . || return 1
    make || return 1
    ./unpackNanoXFast -o unpacked -i test-files.txt
    ./unpackNanoXFast -o unpacked -f 20 -i test-files.txt
    ./unpackNanoXFast -o unpacked -s 2 -b 1 -i test-files.txt
    ./unpackNanoXFast -o unpacked -n 2 -s 10 -f 10 -b 0 -i test-files.txt
}

execute
