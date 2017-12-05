import tensorflow as tf
import numpy as np

def batch_norm(inputs, training, name=None):
    return tf.layers.batch_normalization(
        inputs   = inputs, 
        training = training, 
        name     = name)

def conv2d(inputs, filters, name=None, kernel_size=[3,3], padding="same", activation=tf.nn.elu):
    return tf.layers.conv2d(
        inputs      = inputs,
        filters     = filters,
        kernel_size = kernel_size,
        padding     = padding,
        activation  = activation,
        name        = name)

def maxpool(inputs):
    return tf.layers.max_pooling2d(inputs, pool_size=[2, 2], strides=2)

def dense(inputs, units, name=None, activation=tf.nn.relu):
    return tf.layers.dense(inputs, units=units, activation=activation, name=name)

def flatten(inputs):
    return tf.contrib.layers.flatten(inputs)

def normalize(inputs):
    return tf.nn.l2_normalize(inputs, dim=1)

def conv2d_maxpool(inputs, filters, name=None, kernel_size=[3,3], padding="same", activation=tf.nn.elu):
    return maxpool(conv2d(inputs, filters, name, kernel_size, padding, activation))
