import tensorflow as tf
import numpy as np
from layers import *

def AlexNet(images, out_dim, reuse, training):
    with tf.variable_scope("AlexNet", reuse=reuse):
        l = batch_norm(images, training) # -> 128
        
        l = conv2d(l,  48)
        l = maxpool(l)         # -> 64
        
        l = conv2d(l, 128)
        l = conv2d(l, 128)
        l = maxpool(l)         # -> 32
        
        l = conv2d(l, 192)
        l = conv2d(l, 192)
        l = maxpool(l)         # -> 16
        
        l = conv2d(l, 128)
        l = maxpool(l)         # -> 8
        
        l = flatten(l)         # -> 8192

        l = dense(l, 2048)
        l = dense(l, 2048)
        l = dense(l, out_dim)

        l = normalize(l)
    
    return l

def simple_conv(images, out_dim, reuse, training):
    with tf.variable_scope("simple_conv", reuse=reuse):
        l = batch_norm(images, training) # -> 128

        l = conv2d_maxpool(l,  16) # -> 64
        l = conv2d_maxpool(l,  32) # -> 32
        l = conv2d_maxpool(l,  64) # -> 16
        l = conv2d_maxpool(l, 128) # -> 8
        l = conv2d_maxpool(l, 256) # -> 4

        l = flatten(l)             # -> 1024

        l = dense(l,    1024)
        l = dense(l,    1024)
        l = dense(l, out_dim)
        
        l = normalize(l)
        
        return l

    
def VGG16(images, out_dim, reuse, training):
    with tf.variable_scope("VGG16", reuse=reuse):
        
        l = batch_norm(images, training) # -> 128

        l = conv2d        (l,  64)
        l = conv2d_maxpool(l,  64)      # -> 64
        
        l = conv2d        (l, 128)
        l = conv2d        (l, 128)
        l = conv2d_maxpool(l, 128)      # -> 32
        
        l = conv2d        (l, 512)
        l = conv2d_maxpool(l, 512)      # -> 16
        
        l = conv2d        (l, 512)
        l = conv2d_maxpool(l, 128)      # -> 8
        
        l = flatten(l)                  # -> 8192

        l = dense(l,    2048)
        l = dense(l,    2048)
        l = dense(l, out_dim)
        
        l = normalize(l)
        
        return l

  