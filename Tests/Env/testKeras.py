#!/usr/bin/env python
import tensorflow as tf
import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

print "Keras package" 
print "  Installation: ",os.path.dirname(keras.__file__)
print "  Version: ",keras.__version__
print "Tensorflow package"
print "  Installation: ",os.path.dirname(tf.__file__)
print "  Version: ",tf.__version__

tf.logging.set_verbosity(tf.logging.ERROR)

print "Computing devices:"
from tensorflow.python.client import device_lib
for dev in device_lib.list_local_devices():
    print " -> Devices:",dev.name
    print "    Type:",dev.device_type
    print "    Memory:",dev.memory_limit
    print "    Locality:",dev.locality
    print "    Description:",dev.physical_device_desc
    
    print "    Testing device: ",dev.name,"...",
    with tf.device(dev.name):
        a = tf.constant([1, 2, 3, 4, 5, 6, 7])
        b = tf.constant([2, 3, 4, 5, 6, 7, 8])
        c = tf.add(a,b)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        result = sess.run(c)
        sess.close()
    print "done"
    print

