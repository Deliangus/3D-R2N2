#import pickle
#import numpy
#import os
#from PIL import Image
import tensorflow as tf


#def convnet(x_t, activation, kernel_initializer):
def convnet(x_t):

    n_convfilter = [96, 128, 256, 256, 256, 256]
    n_fc_filters = [1024]
    n_deconvfilter = [128, 128, 128, 64, 32, 2]

    conv_1a = tf.layers.conv2d(x_t, n_convfilter[0], (7, 7))
    conv_1b = tf.layers.conv2d(conv_1a, n_convfilter[0], (3, 3))
    pool_1 = tf.nn.leaky_relu(conv_1b)