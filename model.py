import numpy as np
import tensorflow as tf
from utils import *
import os
from data_process import *

# Refer to https://github.com/trevor-m/tensorflow-SRGAN/blob/master/srgan.py
# Refer to https://github.com/tegg89/SRCNN-Tensorflow/blob/master/model.py 



class SRGenerator:
  def __init__(self, content_loss='mse', use_gan=False, learning_rate=1e-4, num_blocks=16, num_upsamples=2):
      self.learning_rate = learning_rate
      self.num_blocks = num_blocks
      self.num_upsamples = num_upsamples
      self.use_gan = use_gan
      self.reuse_vgg = False
      if content_loss not in ['mse', 'vgg22', 'vgg54']:
          print('Invalid content loss function. Must be \'mse\', \'vgg22\', or \'vgg54\'.')
          exit()
      self.content_loss = content_loss
      #self.device = "/device:"+device+":"+str(device_num)
  
  def _residual_block(self, x, kernel_size, filters, strides=1, training=False):
    #x = tf.nn.relu(x)
    skip = x
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same', use_bias=False)
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same', use_bias=False)
    x = tf.layers.batch_normalization(x, training=training)
    x = x + skip
    return x

  def _Upsample2xBlock(self, x, kernel_size, filters, strides=1):
    """Upsample 2x via SubpixelConv"""
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same', use_bias=False)
    x = tf.depth_to_space(x, 2)
    x = tf.nn.relu(x)
    return x

  def forward(self, x, is_train, reuse):
    """Builds the forward pass network graph"""
#    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
    with tf.variable_scope('generator', reuse=reuse) as scope:
      x = tf.layers.conv2d(x, kernel_size=9, filters=64, strides=1, padding='same')
      x = tf.nn.relu(x)
      #x = tf.contrib.keras.layers.PReLU(shared_axes=[1,2])(x)
      skip = x

      # B x ResidualBlocks
      for i in range(self.num_blocks):
        with tf.name_scope("ResBlock_"+str(i)):
          x = self._residual_block(x, kernel_size=3, filters=64, strides=1, training=is_train)

      x = tf.nn.relu(x)
      x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same', use_bias=False)
      x = tf.layers.batch_normalization(x, training=is_train)
      x = x + skip

      # Upsample blocks
      for i in range(self.num_upsamples-1):
        with tf.name_scope("Upsample_"+str(i)):
          x = self._Upsample2xBlock(x, kernel_size=3, filters=256)
      
      x = tf.layers.conv2d(x, kernel_size=9, filters=3, strides=1, padding='same', name='forward')
    return x
  
  def loss_function(self, y, y_pred):
    """Loss function"""
    # if self.use_gan:
    #   # Weighted sum of content loss and adversarial loss
    #   return self._content_loss(y, y_pred) + 1e-3*self._adversarial_loss(y_pred)
    # Content loss only
    # return self._content_loss(y, y_pred)
    return tf.reduce_mean(tf.square(y- y_pred))
  
  def optimize(self, loss):
    #tf.control_dependencies([discrim_train
    # update_ops needs to be here for batch normalization to work
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
    with tf.control_dependencies(update_ops):
      return tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

  def save(self, sess, saver, checkpoint_dir, step):
    model_name = "SRResNet"
    #model_dir = "%s_%s" % ("srresnet", "valid20")
    #checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

  def load(self, sess, saver, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    #model_dir = "%s_%s" % ("srresnet", "valid20")
    #checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("Restored %s "%ckpt_name)
        
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        return True, int(ckpt_name.split('-')[-1])+1
    else:
        return False, 0
