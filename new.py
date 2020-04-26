from utils import *
from data_process import *
import tensorflow as tf 
import numpy as np

sess = tf.Session()
im = decode_img(tf.read_file('./eb.png'))
im = tf.image.random_crop(im,(128,128,3),seed=1).eval(session=sess)
save_image('./eb64.png',im)

im2 = decode_img(tf.read_file('./eb3.png'))
im2 = tf.image.random_crop(im,(64,64,3),seed=1).eval(session=sess)
save_image('./eb32.png',im)