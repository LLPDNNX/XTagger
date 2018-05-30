import tensorflow as tf
import os

classificationWeights_module = tf.load_op_library(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'libClassificationWeights.so'
))

classification_weights = classificationWeights_module.classification_weights
