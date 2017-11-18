import numpy as np
import os
import cv2
import sys

class Dataset:
    def __init__(self, image_shape):
        self.image_shape  = image_shape
        self.train_images = []
        self.test_images  = []
        self.data_size    = 0

    def load(self, path, data_size):
        files  = os.listdir(path)[:2*data_size]
        file_num = len(files)
        assert file_num >= data_size, "%s %s" % (file_num, data_size)

        h, w, c = self.image_shape
        images = np.zeros([file_num, h, w, c])

        for i,f in enumerate(files):
            f = os.path.join(path, f)
            im = cv2.imread(f)
            if c == 1:
                im = np.max(im, axis=2)
            im = cv2.resize(im, (w,h), interpolation = cv2.INTER_CUBIC).astype('float32')
            im = np.reshape(im, [h,w,c])
            images[i] = im / (im.max()+1e-6)

        self.path         = path
        self.data_size    = data_size
        self.train_images = images[:data_size]
        self.test_images  = images[-data_size:]
    
    def get_next_batch(self, bs):
        data = self.train_images
        idx1 = np.random.choice(np.arange(len(data)), bs)
        idx2 = np.random.choice(np.arange(len(data)), bs)

        img1  = np.copy(data[idx1])
        img2  = np.copy(data[idx2])
        label = np.random.randint(2, size=bs)

        same = np.where(label==1)
        img2[same] = img1[same]

        return img1, img2, label
    

class TripletDataset(Dataset):
    def get_next_batch(self, bs):
        data = self.train_images
        idx = np.random.choice(np.arange(len(data)), bs*2)

        main  = np.copy(data[idx[:bs]])
        same  = np.copy(main)
        diff  = np.copy(data[idx[bs:]])
        
        return main, same, diff


