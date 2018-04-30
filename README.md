# XTagger
Training setup &amp; tools

![status](https://travis-ci.org/LLPDNNX/XTagger.svg?branch=master)

## Setup environment
The following scripts will download [miniconda](https://conda.io/miniconda.html) environment and install the required packages for training networks with [Keras](https://keras.io/), [Tensorflow](https://www.tensorflow.org/) & [ROOT](https://root.cern.ch/).

* for CPU usage only: `source Env/setupEnvCPU.sh <installdir>`
* for CPU & GPU usage: `source Env/setupEnvFull.sh <installdir>`

After the installation the environments can be used with `source Env/env_cpu.sh` or `source Env/env_gpu.sh` respectively.

## Custom operations
A set of custom operation modules for [Tensorflow](https://www.tensorflow.org/) can be installed using [cmake](https://cmake.org/) as following. These allow to train on [ROOT](https://root.cern.ch/) trees directly and to perform preprocessing of the training data such as resampling.
```
mkdir Ops/build
cd Ops/build
cmake ..
make
```
