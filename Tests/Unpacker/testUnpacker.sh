#!/bin/bash

function execute() 
{
    source Env/env_cpu.sh
    cd Unpacker
    cmake . || return 1
    make || return 1
}

execute
