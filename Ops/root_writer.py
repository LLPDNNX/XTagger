import tensorflow as tf
import os

rootwriter_module = tf.load_op_library(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'libRootWriter.so'
))

class root_writer():       
    def __init__(self,
        tensor,
        branches,
        treename,
        filename
    ):
        self._branches = branches
        
        '''
        tensorSplit = []
        for i in range(len(self._branches)):
            tensorSplit.append(tf.slice(tensor,[1,i],[1,1]))
        '''
        self._write_flag = tf.placeholder("int32", [1])
        self._op = rootwriter_module.root_writer(
            tensor, 
            self._write_flag,
            branches,
            treename,
            filename
        )
        
    def write(self):
        return self._op,self._write_flag
        
        
            
