import tensorflow as tf 
import numpy as np
import imageio
from tensorflow.python.ops.image_ops_impl import ResizeMethod


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

def save_image(path, data, highres=True):
    # transform from [-1, 1] to [0, 1]
    if highres:
        data = (data + 1.0) * 0.5
    # transform from [0, 1] to [0, 255], clip, and convert to uint8
    data = np.clip(data * 255.0, 0.0, 255.0).astype(np.uint8)
    imageio.imwrite(path, data)

def bicubic_upsample_x2(images):
    _,h,w,_= tf.shape(images)
    return tf.image.resize(images,[h*2, w*2], method=ResizeMethod.BICUBIC)
    #return NotImplemented
