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
            im = cv2.imread(f).astype('float32')
            if c == 1:
                im = np.max(im, axis=2)
            im = cv2.resize(im, (w,h), interpolation = cv2.INTER_CUBIC)
            im = np.reshape(im, [h,w,c])
            im = im / (im.max()+1e-6)
            images[i] = im.astype('float16')

        self.path         = path
        self.data_size    = data_size
        self.file_num     = file_num
        self.train_images = images[:data_size]
        self.test_images  = images[-data_size:]
    
    def get_next_batch(self, bs, part):
        assert part == 'train' or part == 'test'
        if part == 'train':
            data = self.train_images
        else:
            data = self.test_images

        idx = np.random.choice(np.arange(len(data)), bs*2)

        main  = data[idx[:bs]]
        same  = main
        diff  = data[idx[bs:]]
        
        return main, same, diff
