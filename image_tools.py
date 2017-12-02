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

def __normalize(images):
    for i in range(len(images)):
        im = images[i].astype('float32')/255
        im = im - np.min(im)
        im = im/(np.max(im)+1e-6)
        images[i] = im
        
def resize(images, shape):
    h,w,c = shape
    n     = len(images)
    res   = np.zeros([n,h,w,c], dtype='uint8')
    for i in range(n):
        res[i] = cv2.resize(images[i], (w,h), interpolation = cv2.INTER_CUBIC)
    return res

def plot(images, columns):
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