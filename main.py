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

flags.DEFINE_integer("batch_size", 64, "The size of batch images [128]")
#flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
#flags.DEFINE_integer("label_size", 21, "The size of label to produce [21]")
flags.DEFINE_integer("epochs", 100000, "The number of epochs of learning")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
#flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 2, "The size of scale factor for preprocessing input image [3]")
#flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_string("train_img_dir", "../train_data/DIV2K_train_LR_bicubic/*.png", "Name of train img directory [train_img]")
flags.DEFINE_string("train_label_dir", "../train_data/DIV2K_train_HR/*.png", "Name of train label directory [train_label]")
flags.DEFINE_string("valid_img_dir", "../valid_data/DIV2K_valid_LR_bicubic/*.png", "Name of valid img directory [valid_img]")
flags.DEFINE_string("valid_label_dir", '../valid_data/DIV2K_valid_HR/*.png', "Name of svalid label directory [valid_label]")
flags.DEFINE_string("test_img_dir", "../test_data/DIV2K_valid_LR_bicubic/*.png", "Name of valid img directory [valid_img]")
flags.DEFINE_string("test_label_dir", '../test_data/DIV2K_valid_HR/*.png', "Name of svalid label directory [valid_label]")

flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
flags.DEFINE_string("device", "GPU", "Which device to use")
flags.DEFINE_integer("device_num", 0, "Which device number to use")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def main():
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists('./board'):
        os.makedirs('./board')
    os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.device_num)
    tf.config.set_soft_device_placement(True)
    if True:
    #with tf.device("/device:"+FLAGS.device+":"+str(FLAGS.device_num)):
        model = SRGenerator() 
        #var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        #saver = tf.train.Saver(var_list)
        if FLAGS.is_train:
            train_dataset = make_dataset(FLAGS.train_img_dir, FLAGS.train_label_dir, train=True,batch_size=FLAGS.batch_size)
            train_it = train_dataset.make_initializable_iterator()

            valid_dataset = make_dataset(FLAGS.valid_img_dir, FLAGS.valid_label_dir, train=False, batch_size=1) #dataset을 10개로 자르자
            valid_it = valid_dataset.make_initializable_iterator()
            
            train_x, train_y = train_it.get_next()
            valid_x, valid_y = valid_it.get_next()

            train_pred = model.forward(train_x, True)
            train_loss = model.loss_function(train_y, train_pred)
            train_op = model.optimize(train_loss)

            valid_pred = model.forward(valid_x, False)
            valid_loss = model.loss_function(valid_y, valid_pred) 
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
            saver = tf.train.Saver(var_list)
            with tf.name_scope('train_summary'):
                ts1= tf.summary.image('input_summary', train_x)
                ts2 = tf.summary.image('target_summary', train_y)
                ts3 = tf.summary.image('outputs_summary', train_pred)
            train_img_merged = tf.summary.merge([ts1, ts2, ts3])
            with tf.name_scope('valid_summary'):
                vs1= tf.summary.image('input_summary', valid_x)
                vs2 = tf.summary.image('target_summary', valid_y)
                vs3 = tf.summary.image('outputs_summary', valid_pred)
            valid_img_merged = tf.summary.merge([vs1, vs2, vs3])

            train_loss_avg = tf.placeholder(tf.float32)
            valid_loss_avg = tf.placeholder(tf.float32)
            ts_loss = tf.summary.scalar('train_loss', train_loss_avg)
            vs_loss = tf.summary.scalar('valid_loss', valid_loss_avg) 
            scalar_merged= tf.summary.merge([ts_loss,vs_loss])
            

            with tf.Session() as sess:

                writer = tf.summary.FileWriter('./board/graph', sess.graph)
                writer.add_graph(sess.graph)

                sess.run(tf.global_variables_initializer())
                loaded, start_epoch = model.load(sess, saver, FLAGS.checkpoint_dir)
                if loaded:
                    print(" [*] Load SUCCESS")
                else:
                    print(" [!] Load failed...")
                
                for epoch in tqdm(range(start_epoch,FLAGS.epochs)):
                    start_time = time.time()
                    sess.run(train_it.initializer)
                    t_loss = 0.0
                    count = 0
                    try:
                        while True:
                            # loss = 0
                            # _ = sess.run(train_x)
                            _, loss, train_img_summary = sess.run([train_op, train_loss, train_img_merged])
                            t_loss += loss
                            # print("count : %d"%count)
                            count += 1
                    except tf.errors.OutOfRangeError:
                        pass
                    t_loss /= count
                    sess.run(valid_it.initializer)
                    v_loss = 0.0
                    count = 0
                    while True:
                        try:
                            # _ = sess.run(valid_x)
                            # v_loss = 0
                            loss, valid_img_summary = sess.run([valid_loss, valid_img_merged])
                            v_loss += loss
                            count += 1 
                        except tf.errors.OutOfRangeError:
                            break
                    v_loss /= count
                    print("Epoch: [%2d], time: [%4.4f], train_loss: [%.8f], valid_loss: [%.8f]"% ((epoch+1), time.time()-start_time, t_loss, v_loss))
                    model.save(sess, saver, FLAGS.checkpoint_dir, epoch)
                    
                    summary = sess.run(scalar_merged, feed_dict={train_loss_avg: t_loss, valid_loss_avg: v_loss})
                
                    writer.add_summary(train_img_summary, epoch)
                    writer.add_summary(valid_img_summary, epoch)
                    writer.add_summary(summary, epoch)
            
        else:
            valid_dataset = make_dataset(FLAGS.test_img_dir, FLAGS.test_label_dir, train=False, batch_size=1)
            valid_it = valid_dataset.make_initializable_iterator()
            valid_x, valid_y = valid_it.get_next()
            valid_pred = model.forward(valid_x, False)
            valid_loss = model.loss_function(valid_y, valid_pred) 
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
            saver = tf.train.Saver(var_list)
            with tf.Session() as sess:
                loaded, _ = model.load(sess, saver, FLAGS.checkpoint_dir)
                if loaded:
                    print(" [*] Load SUCCESS")
                else:
                    print(" [!] Load failed...")
                sess.run(valid_it.initializer)
                v_loss = 0.0
                count = 0
                f = open(FLAGS.sample_dir+"/result.txt","w")
                try:
                    while True:
                        start_time = time.time()
                        if count < 10:
                            loss, x, y, pred = sess.run([valid_loss, valid_x, valid_y, valid_pred])
                            bicubic_x = bicubic_upsample_x2(x[0])
                            bicubic_psnr = compute_psnr_np(y[0], bicubic_x)
                            psnr = compute_psnr_np(y, pred)
                            save_image(FLAGS.sample_dir+"/bicubic_"+str(count)+".jpg", bicubic_x)
                            save_image(FLAGS.sample_dir+"/pred_"+str(count)+".jpg", pred[0])
                            v_loss += loss
                            f.write("%dth img => time: [%4.4f], loss: [%.8f], psnr: [%.4f], bicubic_psnr: [%.4f]\n"% ((count), time.time()-start_time, loss, psnr, bicubic_psnr))
                        else:
                            valid_psnr = compute_psnr_tf(valid_y, valid_pred)
                            loss, psnr  = sess.run([valid_loss, valid_psnr])
                            v_loss += loss
                            f.write("%dth img => time: [%4.4f], loss: [%.8f], psnr: [%.4f]\n"% ((count), time.time()-start_time, loss, psnr))
                        count += 1
                except tf.errors.OutOfRangeError:
                    pass
                v_loss /= count
                f.write("Avg. Loss : %.8f"%v_loss)
                f.close()

if __name__=='__main__':
    main()
