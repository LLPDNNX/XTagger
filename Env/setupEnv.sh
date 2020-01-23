#!/bin/bash

SCRIPT_DIR=`dirname ${BASH_SOURCE[0]}`
STOP=""
echo "Script directory "$SCRIPT_DIR
LOGFILE=$SCRIPT_DIR/env.log
echo "Setup run at "`date` > $LOGFILE
echo "Create log file "$LOGFILE

function run_setup()
{
    if [[  -z  $1  ]] ; then
        echo 'Usage:'
        echo '  setupEnv_cpuonly.sh <install_dir>'
        return 1
    fi

    dist=`grep DISTRIB_ID /etc/*-release | awk -F '=' '{print $2}'`

    if [ "$dist" != "Ubuntu" ]; then
        if [[ `uname -r` != *".el7."* ]]; then
            echo "EL 7 required - try different node!, e.g. lx03 (imperial) or lxplus005 (CERN)"
            return 1
        fi
    fi

    INSTALL_DIR=$1

    if [ -d "$1" ]; then
        echo "Error - directory "$INSTALL_DIR" exists!"
        return 1
    fi
    echo "Setting up central environment under "$INSTALL_DIR
    

    mkdir $INSTALL_DIR || return 1
    
    
    INSTALL_ABSDIR=`readlink -e $INSTALL_DIR`

    if [ ! -d "$1" ]; then
        echo "Error - failed to create directory "$INSTALL_ABSDIR"!"
        return 1
    fi
    

    wget -P $INSTALL_ABSDIR https://repo.continuum.io/miniconda/Miniconda2-4.7.12.1-Linux-x86_64.sh &>> $LOGFILE || return 1
    bash $INSTALL_ABSDIR/Miniconda2-4.7.12.1-Linux-x86_64.sh -b -p $INSTALL_ABSDIR/miniconda &>> $LOGFILE || return 1

    CONDA_BIN=$INSTALL_ABSDIR/miniconda/bin
    export PATH=$CONDA_BIN:$PATH
    
    
    export TMPDIR=$INSTALL_ABSDIR/tmp
    export TMPPATH=$TMPDIR
    export TEMP=$TMPDIR
    mkdir $TMPDIR
   
    echo "Creating environment"
    
    conda env create -f $SCRIPT_DIR/environment.yml -q &>> $LOGFILE || return 1
    source activate tf
    conda list
    source deactivate &>> $LOGFILE || return 1
    
    echo "Generate setup script"
    echo "export PATH="$INSTALL_ABSDIR"/miniconda/bin:\$PATH" > $SCRIPT_DIR/env.sh
    echo "source activate tf" >> $SCRIPT_DIR/env.sh
    echo "export KERAS_BACKEND=tensorflow" >> $SCRIPT_DIR/env.sh
    echo "export TF_CPP_MIN_LOG_LEVEL=2" >> $SCRIPT_DIR/env.sh
    echo "export OMP_NUM_THREADS=8 #reduce further if out-of-memory" >> $SCRIPT_DIR/env.sh
}

run_setup $1
if [ $? -eq 0 ]
then
  echo "Successfully setup environment"
else
  tail -n 300 $LOGFILE
  return 1
fi

