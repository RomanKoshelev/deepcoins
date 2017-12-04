import tensorflow as tf
import numpy as np
from layers import *

def simple_conv(images, out_dim, reuse, training):
    use_bn = False
    l = images
    l = conv2d(l,  16, reuse, training, use_bn, pool=True, name='conv_1')
    l = conv2d(l,  32, reuse, training, use_bn, pool=True, name='conv_2')
    l = conv2d(l,  64, reuse, training, use_bn, pool=True, name='conv_3')
    l = conv2d(l, 128, reuse, training, use_bn, pool=True, name='conv_4')
    l = conv2d(l, 256, reuse, training, use_bn, pool=True, name='conv_5')
    l = flatten(l)
    l = dense(l, 4096, reuse, name='fc_1')
    l = dense(l, 4096, reuse, name='fc_2')
    l = dense(l, out_dim, reuse, name='out')
    l = normalize(l)
    return l


def VGG16(images, out_dim, reuse, training):
    use_bn = False
    l = images     # -> 128
    
    l = conv2d(l,  32, reuse, training, use_bn, name='conv_1_1')
    l = conv2d(l,  32, reuse, training, use_bn, name='conv_1_2')
    l = maxpool(l) # -> 64

    l = conv2d(l,  64, reuse, training, use_bn, name='conv_2_1')
    l = conv2d(l,  64, reuse, training, use_bn, name='conv_2_2')
    l = maxpool(l) # -> 32

    l = conv2d(l, 128, reuse, training, use_bn, name='conv_3_1')
    l = conv2d(l, 128, reuse, training, use_bn, name='conv_3_2')
    l = conv2d(l, 128, reuse, training, use_bn, name='conv_3_3')
    l = maxpool(l) # -> 16
    
    l = conv2d(l, 256, reuse, training, use_bn, name='conv_4_1')
    l = conv2d(l, 256, reuse, training, use_bn, name='conv_4_2')
    l = conv2d(l, 256, reuse, training, use_bn, name='conv_4_3')
    l = maxpool(l) # -> 8
    
    l = conv2d(l, 256, reuse, training, use_bn, name='conv_5_1')
    l = conv2d(l, 256, reuse, training, use_bn, name='conv_5_2')
    l = conv2d(l, 256, reuse, training, use_bn, name='conv_5_3')
    l = maxpool(l) # -> 4 x 4 x 256 = 4096
    
    l = flatten(l)
    
    l = dense(l, 4096, reuse, name='fc_1')
    l = dense(l, 4096, reuse, name='fc_2')
    l = dense(l, out_dim, reuse, name='out')
    
    l = normalize(l)
    
    return l


def VGG16_a(images, out_dim, reuse, training):
    use_bn = False
    l = images     # -> 128
    
    l = conv2d(l,  48, reuse, training, use_bn, name='conv_1_1')
    l = conv2d(l,  48, reuse, training, use_bn, name='conv_1_2')
    l = maxpool(l) # -> 64

    l = conv2d(l,  64, reuse, training, use_bn, name='conv_2_1')
    l = conv2d(l,  64, reuse, training, use_bn, name='conv_2_2')
    l = maxpool(l) # -> 32

    l = conv2d(l, 128, reuse, training, use_bn, name='conv_3_1')
    l = conv2d(l, 128, reuse, training, use_bn, name='conv_3_2')
    l = conv2d(l, 128, reuse, training, use_bn, name='conv_3_3')
    l = maxpool(l) # -> 16
    
    l = conv2d(l, 256, reuse, training, use_bn, name='conv_4_1')
    l = conv2d(l, 256, reuse, training, use_bn, name='conv_4_2')
    l = conv2d(l, 256, reuse, training, use_bn, name='conv_4_3')
    l = maxpool(l) # -> 8
    
    l = conv2d(l, 512, reuse, training, use_bn, name='conv_5_1')
    l = conv2d(l, 512, reuse, training, use_bn, name='conv_5_2')
    l = conv2d(l, 512, reuse, training, use_bn, name='conv_5_3')
    l = maxpool(l) # -> 4 x 4 x 512 = 8192
    
    l = flatten(l)
    
    l = dense(l, 4096, reuse, name='fc_1')
    l = dense(l, 4096, reuse, name='fc_2')
    l = dense(l, out_dim, reuse, name='out')
    
    l = normalize(l)
    
    return l


def AlexNet(images, out_dim, reuse, training):
    use_bn = False
    l = images     # -> 128
    
    l = conv2d(l,  48, reuse, training, use_bn, name='conv_1_1')
    l = maxpool(l) # -> 64

    l = conv2d(l, 128, reuse, training, use_bn, name='conv_2_1')
    l = maxpool(l) # -> 32

    l = conv2d(l, 192, reuse, training, use_bn, name='conv_3_1')
    l = conv2d(l, 192, reuse, training, use_bn, name='conv_3_2')
    l = maxpool(l) # -> 16
    
    l = conv2d(l, 128, reuse, training, use_bn, name='conv_3_3')
    l = maxpool(l) # -> 8 x 8 x 128 = 8192
    
    l = flatten(l)
    
    l = dense(l, 2048, reuse, name='fc_1')
    l = dense(l, 2048, reuse, name='fc_2')
    l = dense(l, out_dim, reuse, name='out')
    
    l = normalize(l)
    
    return l