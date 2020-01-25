'''===================================================================
Copyright 2019 Matthias Komm, Vilius Cepaitis, Robert Bainbridge, 
Alex Tapper, Oliver Buchmueller. All Rights Reserved. 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
    
Unless required by applicable law or agreed to in writing, 
software distributed under the License is distributed on an "AS IS" 
BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express 
or implied.See the License for the specific language governing 
permissions and limitations under the License.
==================================================================='''



import tensorflow as tf
import os

rootreader_module = tf.load_op_library(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'libRootReader.so'
))

class root_reader():
    @staticmethod
    def slice_and_reshape(start,size,shape=None):
        if shape==None:
            return lambda tensor: tensor[:,start:start+size]
        else:
            return lambda tensor: tf.transpose(tf.reshape(tensor[:,start:start+size],shape=shape),perm=[0,2,1])
        
    def __init__(self,
        queue,
        feature_dict,
        treename,
        batch=1,
        naninf=0
    ):
        self._feature_dict = feature_dict
        
        self._branch_list = []
        self._output_formatters = {}
        index = 0
        for feature_name in sorted(self._feature_dict.keys()):
            
            feature_values = self._feature_dict[feature_name]
            if not feature_values.has_key("max"):
                self._output_formatters[feature_name]=root_reader.slice_and_reshape(
                    index,
                    len(feature_values["branches"])
                )
                index+=len(feature_values["branches"])
                self._branch_list.extend(feature_values["branches"])
                
            else:
                self._output_formatters[feature_name]=root_reader.slice_and_reshape(
                    index,
                    len(feature_values["branches"])*feature_values["max"],
                    [-1,len(feature_values["branches"]),feature_values["max"]]
                )
                index+=len(feature_values["branches"])*feature_values["max"]
                for branch_name in feature_values["branches"]:
                    self._branch_list.append(
                        branch_name+"["+str(feature_values["max"])+"]"
                    )
                 
                
        self._op_batch, self._op_num = rootreader_module.root_reader(
            queue.queue_ref, 
            self._branch_list,
            treename=treename,
            naninf=naninf, 
            batch=batch
        )
        
    def raw(self):
        return {"raw":self._op_batch,"num":self._op_num}
        
    def batch(self,preprocess=lambda x:x):
        result = {}
        op_batch_preprocessed = preprocess(self._op_batch)
        for feature_name in sorted(self._output_formatters.keys()):
            result[feature_name]=self._output_formatters[feature_name](op_batch_preprocessed)
        result["num"] = self._op_num
        return result
        
            
