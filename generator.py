import numpy as np
import os
from tqdm import tqdm
import image_tools as imt

class Generator:
    def __init__(self):
        self.ethalons  = None
        self.augmented = None
    
    def generate(self, images_path, image_shape, augmentator, image_num, aug_num):
        h,w,c = image_shape
        ethalons  = imt.load(images_path, image_num)
        ethalons  = imt.resize(ethalons, [h,w,3])
        image_num = image_num or len(ethalons)
        augmented = np.zeros([image_num, aug_num, h, w, 1], dtype='uint8')
        for i in tqdm(range(aug_num)):
            res = augmentator.augment(ethalons)
            res = np.mean(res, axis=3, keepdims=True)
            augmented[:,i,:,:] = res
        augmented[:,0,:] = np.mean(ethalons, axis=3, keepdims=True)
        self.augmented = augmented
        return self.augmented
    
    def save(self, dataset_dir):
        data = self.augmented
        ds_name = '_'.join(['%d'%d for d in data.shape])+'_%s'%data.dtype
        ds_path = os.path.join(dataset_dir, ds_name+'.npy')
        np.save(ds_path, data)
        return ds_path