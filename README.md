# XTagger
Training setup &amp; tools

![status](https://travis-ci.org/LLPDNNX/XTagger.svg?branch=master)

## Setup environment
The script `source Env/setupEnvCPU.sh <installdir>` will download 
[miniconda](https://conda.io/miniconda.html) environment and install the
required packages for training networks with [Keras](https://keras.io/), 
[Tensorflow](https://www.tensorflow.org/) & [ROOT](https://root.cern.ch/).

After the installation the environments can be used with `source Env/env_cpu.sh` 
or `source Env/env_gpu.sh` respectively.

## Custom operations
A set of custom operation modules for [Tensorflow](https://www.tensorflow.org/) 
can be installed using [cmake](https://cmake.org/) as following. These allow to 
train on [ROOT](https://root.cern.ch/) trees directly and to perform preprocessing 
of the training data such as resampling.

```
mkdir Ops/build
cd Ops/build
cmake .. -DCMAKE_INSTALL_PREFIX=<installdir>
make install
```

## Training using nanoX unpacked samples:

Training is be performed by unpacking jets in extended nanoAOD 
([LLPReco](https://github.com/LLPDNNX/LLPReco)) samples.

```
python Training/training.py \
--gpu \
-b 10000 \
--train samples/nanox_ctau_10_train.txt \
--test samples/nanox_ctau_10_test.txt \
-e 100 \
--name ctau_10 \
-c \
-n 10 \
--name ctau_10
```

This would start training using GPUs if available, with a batch size of 10000 
and with specified training and testing samples. Furthermore, the training would 
be performed for 100 epochs and achieving balance of all classes as well as 
kinematic resampling. The output folder will be created as output/ctau_10.
To load a different model, the ``-m`` parameter can be used.

