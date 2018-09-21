import argparse
import os
import sys
import tensorflow as tf
import keras
import numpy
from keras import backend as K
from feature_dict import featureDict
import llp_model_da

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--parametric', action='store_true',
                    dest='parametric',
                    help='train a parametric model', default=False)
parser.add_argument('weightfile', metavar='weightfile', type=str, nargs=1,
                    help='a weight file from keras')
arguments = parser.parse_args()


if not os.path.exists(arguments.weightfile[0]):
    print "Error - weight file '",arguments.weightfile[0],"' does not exists"
    sys.exit(1)
if not arguments.weightfile[0].endswith(".hdf5"):
    print "Error - file '",parser.weightfile[0],"' is not a hdf5 file"
    sys.exit(1)

sess = K.get_session()

tf_cpf = tf.placeholder('float32',shape=(None,featureDict["cpf"]["max"],len(featureDict["cpf"]["branches"])),name="cpf")
cpf = keras.layers.Input(tensor=tf_cpf)
tf_npf = tf.placeholder('float32',shape=(None,featureDict["npf"]["max"],len(featureDict["npf"]["branches"])),name="npf")
npf = keras.layers.Input(tensor=tf_npf)
tf_sv = tf.placeholder('float32',shape=(None,featureDict["sv"]["max"],len(featureDict["sv"]["branches"])),name="sv")
sv = keras.layers.Input(tensor=tf_sv)
tf_globalvars = tf.placeholder('float32',shape=(None,len(featureDict["globalvars"]["branches"])),name="globalvars")
globalvars = keras.layers.Input(tensor=tf_globalvars)
if arguments.parametric:
    tf_gen = tf.placeholder('float32',shape=(None,1),name="gen")
    gen = keras.layers.Input(tensor=tf_gen)
else:
    gen = None

print "cpf shape: ",cpf.shape.as_list()
print "npf shape: ",npf.shape.as_list()
print "sv shape: ",sv.shape.as_list()
print "globalvars shape: ",globalvars.shape.as_list()
if arguments.parametric:
    print "gen shape: ",gen.shape.as_list()
    
modelDA = llp_model_da.ModelDA(
    len(featureDict["truth"]["branches"]),
    isParametric=arguments.parametric,
    useLSTM=False
)


print "learning phase: ",sess.run(keras.backend.learning_phase())

class_prediction = modelDA.predictClass(globalvars,cpf,npf,sv,gen)
prediction = tf.identity(class_prediction,name="prediction")

#prediction = tf.identity(full_prediction,name="prediction")

if arguments.parametric:
    model = keras.Model(inputs=[gen, globalvars, cpf, npf, sv], outputs=class_prediction)
else:
    model = keras.Model(inputs=[globalvars, cpf, npf, sv], outputs=class_prediction)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

def shape(tf_tensor):
    dims = tf_tensor.shape.as_list()
    dims[0] = 1
    return dims

model.load_weights(arguments.weightfile[0])




#test if graph can be executed
feed_dict={
    tf_globalvars:numpy.zeros(shape(tf_globalvars)),
    tf_cpf:numpy.zeros(shape(tf_cpf)),
    tf_npf:numpy.zeros(shape(tf_npf)),
    tf_sv:numpy.zeros(shape(tf_sv)),
}
if arguments.parametric:
    feed_dict[tf_gen] = numpy.zeros(shape(tf_gen))

full_prediction_val = sess.run(
    class_prediction,
    feed_dict=feed_dict
)

const_graph = tf.graph_util.convert_variables_to_constants(
    sess,
    sess.graph.as_graph_def(),
    ["prediction"]
)
tf.train.write_graph(const_graph,"",arguments.weightfile[0].replace("hdf5","pb"),as_text=False)

print "Sucessfully saved graph and weights into '%s'"%arguments.weightfile[0].replace("hdf5","pb")

