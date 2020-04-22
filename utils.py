import tensorflow as tf 
import numpy as np

def conv2d(input, kernel_size=3, input_filters=64, output_filters=64, strides=1, padding='SAME'):
    tf.nn.conv2d()
    return tf.nn.conv2d(input, filters=[kernel_size, kernel_size, input_filters,output_filters], 
    strides=[1,strides,strides,1], padding=padding)

def compute_psnr(ref, target):
    ref = tf.cast(ref, tf.float32)
    target = tf.cast(target, tf.float32)
    diff = target - ref
    sqr = tf.multiply(diff, diff)
    err = tf.reduce_sum(sqr)
    v = tf.shape(diff)[0] * tf.shape(diff)[1] * tf.shape(diff)[2] * tf.shape(diff)[3]
    mse = err / tf.cast(v, tf.float32)
    psnr = 10. * (tf.log(255. * 255. / mse) / tf.log(10.))

    return psnr

def save_image(path, data, highres=False):
  # transform from [-1, 1] to [0, 1]
  if highres:
    data = (data + 1.0) * 0.5
  # transform from [0, 1] to [0, 255], clip, and convert to uint8
  data = np.clip(data * 255.0, 0.0, 255.0).astype(np.uint8)
  misc.toimage(data, cmin=0, cmax=255).save(path)

def save_bicubic()