import tensorflow as tf
import numpy as np
import os 
import glob 



train_img_data_dir = '../DIV2K_train_LR_bicubic/*.png'
train_label_data_dir = '../DIV2K_train_HR/*.png'
valid_img_data_dir = '../DIV2K_valid_LR_bicubic/*.png'
valid_label_data_dir = '../DIV2K_valid_HR/*.png'

#train_img_data_list = os.listdir(train_img_data_dir)
#train_label_data_list = os.listdir(train_label_data_dir)
#print(train_img_data_list)
#print(train_label_data_list)

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_png(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32) 
  # convert range into [-1, 1]
  img = img*2. - 1.
  return img
  
def process_path(img_file_path, label_file_path):
  #img_file_path, label_file_path = file_path
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


seed = 0
def random_crop_size64(images, labels):
  global seed
  seed += 1
  return (tf.image.random_crop(images,[32,32,3],seed=seed),tf.image.random_crop(labels,[64,64,3],seed=seed))

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.

def random_crop_size96(images, labels):
  global seed
  seed += 1
  return (tf.image.random_crop(images,[48,48,3],seed=seed),tf.image.random_crop(labels,[96,96,3],seed=seed))


def make_dataset(img_file_path, label_file_path, train=True, batch_size=128):
  ds1 = tf.data.Dataset.list_files(img_file_path, shuffle=False)
  ds2 = tf.data.Dataset.list_files(label_file_path, shuffle=False)
  ds = tf.data.Dataset.zip((ds1, ds2))
  labeled_ds = ds.map(process_path, num_parallel_calls=10)
  if train:
    train_labeled_ds = labeled_ds.map(random_crop_size64).shuffle(800).batch(batch_size).prefetch(buffer_size=5)
    return train_labeled_ds

  else:
    valid_labeled_ds = labeled_ds.batch(1).prefetch(buffer_size=5)
    return valid_labeled_ds
'''
ds= make_dataset(train_img_data_dir,train_label_data_dir, batch_size=64)
it = ds.make_initializable_iterator()
n1, n2 = it.get_next()
with tf.Session() as sess:
  
  for i in range(2):
    sess.run(it.initializer)
    while True:
      try:
        print(sess.run([tf.shape(n1), tf.shape(n2)]))
      except tf.errors.OutOfRangeError:
        print("end")
        break
'''