import tensorflow as tf
import numpy as np

def conv2d(inputs, filters, reuse, training, bn, name, kernel_size=[3,3], pool=False, padding="same"):
    l = tf.layers.conv2d(
        inputs      = inputs,
        filters     = filters,
        kernel_size = kernel_size,
        padding     = padding,
        activation  = None if bn else tf.nn.relu,
        reuse       = reuse,
        name        = name)
    if bn:
        l = tf.layers.batch_normalization(
            inputs   = l, 
            reuse    = reuse,
            training = training, 
            name     = name+'_bn')
        l = tf.nn.elu(l)
    if pool:
        l = maxpool(l)
    return l

def maxpool(inputs):
    return tf.layers.max_pooling2d(inputs, pool_size=[2, 2], strides=2)

def dense(inputs, units, reuse, name, activation=tf.nn.relu):
    return tf.layers.dense(inputs, units=units, activation=activation, reuse=reuse, name=name)

def flatten(inputs):
    return tf.contrib.layers.flatten(inputs)

def normalize(inputs):
    return tf.nn.l2_normalize(inputs, dim=1)
