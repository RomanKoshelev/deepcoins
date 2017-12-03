import numpy as np
import os
import cv2

class Dataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data      = None 

    def load(self):
        self.data = np.load(self.data_path)
    
    def get_next_batch(self, bs):
        n,a = self.data.shape[:2]
        assert 2*bs<=n and bs<=a
        idx_n   = np.random.choice(np.arange(n), bs*2, replace=False)
        idx_a_1 = np.random.choice(np.arange(a), bs,   replace=False)
        idx_a_2 = np.random.choice(np.arange(a), bs,   replace=False)
        idx_a_3 = np.random.choice(np.arange(a), bs,   replace=False)

        main  = self.data[idx_n[:bs],  idx_a_1]
        same  = self.data[idx_n[:bs],  idx_a_2]
        diff  = self.data[idx_n[-bs:], idx_a_3]
        
        return main, same, diff
