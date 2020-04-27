import tensorflow as tf 
import numpy as np
import imageio
import cv2
from math import sqrt, log10
from tensorflow.python.ops.image_ops_impl import ResizeMethod


def compute_psnr_np(ref, target):
    #input is numpy array with range [-1,1]
    mse = np.mean((ref-target) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 2
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def compute_psnr_tf(ref,target):
    ref = tf.cast(ref, tf.float32)
    target = tf.cast(target, tf.float32)
    diff = target - ref
    sqr = tf.multiply(diff, diff)
    err = tf.reduce_sum(sqr)
    v = tf.shape(diff)[0] * tf.shape(diff)[1] * tf.shape(diff)[2] * tf.shape(diff)[3]
    mse = err / tf.cast(v, tf.float32)
    psnr = 20. * (tf.log(2. / mse) / tf.log(10.))

    return psnr

def save_image(path, data, highres=True):
    #input is numpy 3D array
    # transform from [-1, 1] to [0, 1]
    if highres:
        data = (data + 1.0) * 0.5
    # transform from [0, 1] to [0, 255], clip, and convert to uint8
    data = np.clip(data * 255.0, 0.0, 255.0).astype(np.uint8)
    imageio.imwrite(path, data)

def bicubic_upsample_x2(images):
    #input is numpy 3D array
    dim = (images.shape[1]*2, images.shape[0]*2)
    return cv2.resize(images, dim, interpolation=cv2.INTER_CUBIC)
    #return NotImplement
