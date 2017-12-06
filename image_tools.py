import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def load(path, num=None):
    files  = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if num is None:
        num = len(files)
    images = []
    for i,f in enumerate(files[:num]):
        f = os.path.join(path, f)
        im = cv2.imread(f)
        images.append(im)
    return images

def normalize(images):
    res = np.zeros_like(images)
    for i in range(len(images)):
        res[i] = images[i]
        im = res[i]
        im = im - np.min(im)
        im = im/(np.max(im)+1e-6)
    return res
        
def resize(images, shape):
    h,w,c = shape
    n     = len(images)
    res   = np.zeros([n,h,w,c], dtype='uint8')
    for i in range(n):
        res[i] = cv2.resize(images[i], (w,h), interpolation = cv2.INTER_CUBIC)
    return res

def plot(images, columns):
    if images.shape[-1] == 1:
        images = np.tile(images, [1,1,1,3])*255.
        images = images.astype('uint8')
    
    num  = len(images)
    rows = max(1,num//columns)
    W    = 18
    plt.figure(figsize=(W,W*rows/columns))
    for i in range(len(images)):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.axis("off")
        im = images[i]
        if len(im.shape)==3:
            if im.shape[2]==1:
                im = np.reshape(im, im.shape[:2])
        if len(im.shape)==2:
            plt.imshow(im, cmap='gray')
        else:
            plt.imshow(im)
            
def grayscale(images):
    images = np.mean(images, axis=3, keepdims=True)/255
    return images.astype('float16')

def choice(images, num):
    return images[np.random.choice(images.shape[0], num, replace=False), :]

def uint8_to_float32(images):
    return images.astype('float32')/256