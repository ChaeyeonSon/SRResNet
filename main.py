import tensorflow as tf 
import numpy as np
import argparse
from data_process import *
from model import *

import os
import pprint
from tqdm import tqdm
import time

flags = tf.app.flags

flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 21, "The size of label to produce [21]")
flags.DEFINE_integer("epochs", 100000, "The number of epochs of learning")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
flags.DEFINE_integer("gpu", 0, "Which GPU to use")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def train():
    return NotImplemented

def test():
    return NotImplemented

def main():
    #pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists('./board'):
        os.makedirs('./board')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    print("Hi!")
    model = SRGenerator(training=FLAGS.is_train)
    print("Hi?")
    if FLAGS.is_train:
        
        train_dataset = make_dataset(True,FLAGS.batch_size)
        train_it = train_dataset.make_initializable_iterator()

        valid_dataset = make_dataset(False, batch_size=1) #dataset을 10개로 자르자
        valid_it = valid_dataset.make_initializable_iterator()
        
        train_x, train_y = train_it.get_next()
        valid_x, valid_y = valid_it.get_next()

        train_pred = model.forward(train_x)
        train_loss = model.loss_function(train_y, train_pred)
        train_op = model.optimize(train_loss)

        valid_pred = model.forward(valid_x)
        valid_loss = model.loss_function(valid_y, valid_pred) 

        with tf.name_scope('inputs_summary'):
            tf.summary.image('input_summary', train_x)

        with tf.name_scope('targets_summary'):
            tf.summary.image('target_summary', train_y)

        with tf.name_scope('outputs_summary'):
            tf.summary.image('outputs_summary', train_pred)

        train_loss_avg = tf.placeholder(tf.float32)
        valid_loss_avg = tf.placeholder(tf.float32)
        tf.summary.scalar('train_loss', train_loss_avg)
        tf.summary.scalar('valid_loss', valid_loss_avg)
        # tf.summary.scalar('PSNR', psnr)
        # tf.summary.scalar('learning_rate', Net.learning_rate)
        merged = tf.summary.merge_all()


        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./board/graph', sess.graph)
            writer.add_graph(sess.graph)

            sess.run(tf.global_variables_initializer())
            model.sess = sess # 이거 완전 악순데...
            if model.load(FLAGS.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
            sess.run(train_it.initializer)
            for epoch in tqdm(range(FLAGS.epochs)):
                start_time = time.time()
                t_loss = 0.0
                count = 0
                try:
                    while True:
                        _, loss = sess.run(train_op, train_loss)
                        t_loss += loss
                        count += 1
                except tf.errors.OutOfRangeError:
                    pass
                t_loss /= count
                sess.run(valid_it.initializer)
                v_loss = 0.0
                for i in range(10):
                    v_loss += sess.run(valid_loss)/10
                print("Epoch: [%2d], time: [%4.4f], train_loss: [%.8f], valid_loss: [%.8f]"% ((epoch+1), time.time()-start_time, t_loss, v_loss))
                model.save(FLAGS.checkpoint_dir, epoch)
                summary = sess.run(merged, feed_dict={train_loss_avg: t_loss, valid_loss_avg: v_loss})
                writer.add_summary(summary, epoch)
        
    else:
        valid_dataset = make_dataset(False, batch_size=1)
        valid_it = valid_dataset.make_initializable_iterator()
        valid_pred = model.forward(valid_x)
        valid_loss = model.loss_function(valid_y, valid_pred) 
        with tf.Session() as sess:
            model.sess=sess
            if model.load(FLAGS.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
            sess.run(valid_it.initializer)
            v_loss = 0.0
            count = 0
            try:
                while True:
                    start_time = time.time()
                    loss = sess.run(valid_loss)
                    v_loss += loss
                    if count < 10:
                        save_image(FLAGS.sample_dir+"/"+str(count)+".jpg", valid_pred)
                        psnr = compute_psnr(valid_y, valid_pred)
                        save_bicubic
                    count += 1
                    f.write("Epoch: [%2d], time: [%4.4f], loss: [%.8f], psnr: [%.4f]"% ((epoch+1), time.time()-start_time, loss, psnr))
            
            except tf.errors.OutOfRangeError:
                pass
            v_loss /= count

if __name__=='__main__':
    main()
