import tensorflow as tf

resampler_module = tf.load_op_library('Ops/libResampler.so')

class resampler():       
    def __init__(self,
        rates,
        batch
    ):
        if type(batch)==type(dict()):
            self.inputBatch = []
            for name in sorted(batch.keys()):
                self.inputBatch.append(batch[name])
     
        elif type(batch)==type(list()):
            self.inputBatch = batch


        output = resampler_module.resampler(
            rates,
            self.inputBatch
        )
        
        
        if type(batch)==type(dict()):
            self.outputBatch = {}
            for i,name in enumerate(sorted(batch.keys())):
                self.outputBatch[name]=output[i]
        elif type(batch)==type(list()):
            self.outputBatch = output
            
    def resample(self):
        return self.outputBatch

            
