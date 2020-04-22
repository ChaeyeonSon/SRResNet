import tensorflow as tf
import numpy as np
import os 
from glob import glob 

# random crop
# tf.image
# tf.dataset
# from generator

# train_img_data_list = glob('data\\mnist_png\\training\\*\\*.png')
# train_label_data_list = glob('data\\mnist_png\\training\\*\\*.png')
# valid_img_data_list = glob('data\\mnist_png\\training\\*\\*.png')
# valid_label_data_list = glob('data\\mnist_png\\training\\*\\*.png')

train_img_data_dir = '..\\DIV2K_train_LR_bicubic\\*.png'
train_label_data_dir = '..\\DIV2K_train_HR\\*.png'
valid_img_data_dir = '..\\DIV2K_valid_LR_bicubic\\*.png'
valid_label_data_dir = '..\\DIV2K_valid_HR\\*.png'


def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_png(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32) 
  # convert range into [-1, 1]
  img = img*2. - 1.
  return img
  
def process_path(file_path):
  img_file_path, label_file_path = file_path[0], file_path[1]
  label = tf.io.read_file(label_file_path)
  label = decode_img(label)
  # load the raw data from the file as a string
  img = tf.io.read_file(img_file_path)
  img = decode_img(img)
  return img, label

def random_crop_and_pad_image_and_labels(image, labels, size):
  """Randomly crops `image` together with `labels`.

  Args:
    image: A Tensor with shape [D_1, ..., D_K, N]
    labels: A Tensor with shape [D_1, ..., D_K, M]
    size: A Tensor with shape [K] indicating the crop size.
  Returns:
    A tuple of (cropped_image, cropped_label).
  """
  combined = tf.concat([image, labels], axis=2)
  image_shape = tf.shape(image)
  combined_pad = tf.image.pad_to_bounding_box(
      combined, 0, 0,
      tf.maximum(size[0], image_shape[0]),
      tf.maximum(size[1], image_shape[1]))
  last_label_dim = tf.shape(labels)[-1]
  last_image_dim = tf.shape(image)[-1]
  combined_crop = tf.random_crop(
      combined_pad,
      size=tf.concat([size, [last_label_dim + last_image_dim]],
                     axis=0))
  return (combined_crop[:, :, :last_image_dim],
          combined_crop[:, :, last_image_dim:])

def random_crop_size64(images):
  return random_crop_and_pad_image_and_labels(images[0],images[1], [64,64])
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.

def make_dataset(train=True, batch_size=128):
  if train:
    train_list_ds = tf.data.Dataset.list_files([train_img_data_dir, train_label_data_dir])
    train_labeled_ds = train_list_ds.map(process_path, num_parallel_calls=10).map(random_crop_size64)
    train_labeled_ds = train_labeled_ds.shuffle(1000).batch(batch_size).prefetch(buffer_size=10)
    return train_labeled_ds

  else:
    valid_list_ds = tf.data.Dataset.list_files([valid_img_data_dir, valid_label_data_dir])
    valid_labled_ds = valid_list_ds.map(process_path, num_parallel_calls=10).batch(1).prefetch(buffer_size=10)
    return valid_labled_ds
