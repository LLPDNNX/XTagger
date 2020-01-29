import logging

class Devices():
    def __init__(self):
        self.nCPU = 0
        self.nGPU = 0
        self.nUnknown = 0
        from tensorflow.python.client import device_lib
        for dev in device_lib.list_local_devices():
            if dev.device_type=="CPU":
                self.nCPU+=1
            elif dev.device_type=="GPU":
                self.nGPU+=1
            else:
                self.nUnknown+=1
        logging.info("Found %i/%i/%i (CPU/GPU/unknown) devices"%(self.nCPU,self.nGPU,self.nUnknown))

    def nCPU(self):
        return self.nCPU
        
    def nGPU(self):
        return self.nGPU
        
    def nUnknown(self):
        return self.nUnknown
