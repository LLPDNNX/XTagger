import tensorflow as tf
import logging
import xtagger
import os

class Pipeline():
    def __init__(
        self,
        files, 
        features, 
        labelNameList,
        weightFile,
        batchSize, 
        resample=True,
        repeat=1,
        bagging=1.,
        maxThreads = 6
    ):
        self.files = files
        self.features = features
        self.labelNameList = labelNameList
        self.weightFile = weightFile
        self.batchSize = batchSize
        self.resample = resample
        self.repeat = repeat
        self.bagging = bagging
        self.maxThreads = maxThreads
    

    def init(self,isLLPFct):
        with tf.device('/cpu:0'):
            if self.bagging>0. and self.bagging<1.:
                inputFileList = random.sample(self.files,int(max(1,round(len(self.files)*self.bagging))))
            else:
                inputFileList = self.files
            fileListQueue = tf.train.string_input_producer(
                    inputFileList, num_epochs=self.repeat, shuffle=True)

            rootreader_op = []
            resamplers = []
            OMP_NUM_THREADS = -1
            if os.environ.has_key('OMP_NUM_THREADS'):
                try:
                    OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
                except Exception:
                    pass
            if OMP_NUM_THREADS>0:
                self.maxThreads = min(OMP_NUM_THREADS,self.maxThreads)
            
            for _ in range(min(1+int(len(inputFileList)/2.), self.maxThreads)):
                reader_batch = max(10,int(self.batchSize/(2.5*self.maxThreads)))
                reader = xtagger.root_reader(fileListQueue, self.features, "jets", batch=reader_batch).batch()
                rootreader_op.append(reader)
                if self.resample:
                    weight = xtagger.classification_weights(
                        reader["truth"],
                        reader["globalvars"],
                        self.weightFile,
                        self.labelNameList,
                        [0, 1]
                    )
                    resampled = xtagger.resampler(
                        weight,
                        reader
                    ).resample()

                    resamplers.append(resampled)

            minAfterDequeue = self.batchSize * 2
            capacity = minAfterDequeue + 3*self.batchSize
            batch = tf.train.shuffle_batch_join(
                resamplers if self.resample else rootreader_op,
                batch_size=self.batchSize,
                capacity=capacity,
                min_after_dequeue=minAfterDequeue,
                enqueue_many=True  # requires to read examples in batches!
            )
            if self.resample:
                isSignal = isLLPFct(batch)
                batch["gen"] = xtagger.fake_background(batch["gen"], isSignal, 0)

            return batch


