import tensorflow as tf
import os

fakebackground_module = tf.load_op_library(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'libFakeBackground.so'
))

fake_background = fakebackground_module.fake_background
