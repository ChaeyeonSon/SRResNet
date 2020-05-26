import tensorflow as tf 
import numpy as np
import imageio
import cv2
from math import sqrt, log10
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from scipy import signal
from scipy import ndimage


def save_image(path, data, highres=True):
    #input is numpy 3D array
    # transform from [-1, 1] to [0, 1]
    if highres:
        data = (data + 1.0) * 0.5
    # transform from [0, 1] to [0, 255], clip, and convert to uint8
    data = np.clip(data * 255.0, 0.0, 255.0).astype(np.uint8)
    imageio.imwrite(path, data)

def bicubic_upsample_x2_np(images):
    #input is numpy 3D array
    dim = (images.shape[1]*2, images.shape[0]*2)
    return cv2.resize(images, dim, interpolation=cv2.INTER_CUBIC)
    #return NotImplement

def bicubic_upsample_x2_tf(images):
    size = [tf.shape(images)[1]*2, tf.shape(images)[2]*2]
    return tf.image.resize(images, size, method=ResizeMethod.BICUBIC)


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
#    ref = tf.cast(ref, tf.float32)
#    target = tf.cast(target, tf.float32)
#    diff = target - ref
#    sqr = tf.multiply(diff, diff)
#    err = tf.reduce_sum(sqr)
#    v = tf.shape(diff)[0] * tf.shape(diff)[1] * tf.shape(diff)[2] * tf.shape(diff)[3]
#    mse = err / tf.cast(v, tf.float32)
#    psnr = 20. * (tf.log(2. / mse) / tf.log(10.))

#    psnr = tf.image.psnr(ref[0],target[0],max_val=2.0)

    psnr = tf.image.psnr(ref[0],target[0],max_val=2.0)
    return psnr

from skimage.measure import compare_ssim as ssim

def compute_ssim_np(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = gauss.fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 2.0
#    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))
    """
    return ssim(img1, img2, data_range=2.0,multichannel=True)


def compute_ssim_tf(img1, img2):
    #return tf.image.ssim(img1,img2, max_val=2.0)
    return tf.image.ssim(img1,img2, max_val=2.0)

