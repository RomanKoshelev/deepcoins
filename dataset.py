import numpy as np
import os
import cv2
import image_tools as imt

class Dataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data      = None 

    def load(self):
        self.data = np.load(self.data_path)
    
    def get_next_batch(self, bs):
        n,a = self.data.shape[:2]
        
        to_float = imt.uint8_to_float32
        
        i_m = np.random.choice(np.arange(n), bs, replace=bs>n)
        i_d = np.random.choice(np.arange(n), bs, replace=bs>n)
        
        a_m = np.random.choice(np.arange(a), bs, replace=bs>a)
        a_s = np.random.choice(np.arange(a), bs, replace=bs>a)
        a_d = np.random.choice(np.arange(a), bs, replace=bs>a)

        main  = to_float(self.data[i_m, a_m])
        same  = to_float(self.data[i_m, a_s])
        diff  = to_float(self.data[i_d, a_d])
            
        return main, same, diff

    def get_ethalons(self):
        return imt.uint8_to_float32(self.data[:,0])

    def get_augmented(self):
        a = self.data.shape[1]
        r = np.random.randint(a-1)+1
        return imt.uint8_to_float32(self.data[:,r])
    
    def get_data_size(self):
        return self.data.shape[0]*self.data.shape[1]

    
def load_datasets(*paths):
    datasets = []
    for path in paths:
        ds   = Dataset(path)
        ds.load()
        print("%s:\n" % path, list(ds.data.shape), ds.data.dtype)
        datasets.append(ds)        
    return datasets