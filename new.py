import tensorflow as tf 
import numpy as np 

# dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset = tf.data.Dataset.range(100)
it =dataset.make_initializable_iterator()
# it = dataset.make_one_shot_iterator()
a = it.get_next()

with tf.Session() as sess:
    sess.run(it.initializer)
    while True:
        try:
            # print(a)
            print(sess.run(a))
        except tf.errors.OutOfRangeError:
            print('end\n')
            break
