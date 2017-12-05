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
        assert 2*bs<=n, "Too large batch size %d"%bs
        idx_n   = np.random.choice(np.arange(n), bs*2, replace=False)
        idx_a_1 = np.random.choice(np.arange(a), bs)
        idx_a_2 = np.random.choice(np.arange(a), bs)
        idx_a_3 = np.random.choice(np.arange(a), bs)

        to_float = imt.uint8_to_float32
        main  = to_float(self.data[idx_n[:bs],  idx_a_1])
        same  = to_float(self.data[idx_n[:bs],  idx_a_2])
        diff  = to_float(self.data[idx_n[-bs:], idx_a_3])
        
        return main, same, diff

    def get_ethalons(self):
        return imt.uint8_to_float32(self.data[:,0])

    def get_augmented(self):
        a = self.data.shape[1]
        r = np.random.randint(a-1)+1
        return imt.uint8_to_float32(self.data[:,r])
    
    def get_data_size(self):
        return self.data.shape[0]*self.data.shape[1]
