#
# Copyright 2016 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import print_function

import time
import os.path
import tensorflow as tf
import neuralnetwork as nn
import numpy as np
import sys
import argparse

import json

sys.path.append('../util')
import netFunctions as nf

tf.logging.set_verbosity(tf.logging.DEBUG)

GREEN = "\033[0;32m"
NC = "\033[0m"

FLAGS = None
check_save = None
cluster_spec = None

z_dimensions = 112

def writeTxt(dloss, gloss, duration):
    try:
        f = open(info_file, 'ab')

        f.write('learning_rate :' + FLAGS.learning_rate.__str__() + ',\n')
        f.write('decay_steps :' + FLAGS.decay_steps.__str__() + ',\n')
        f.write('num_epochs :' + FLAGS.num_epochs.__str__() + ',\n')
        f.write('batch_size :' + FLAGS.batch_size.__str__() + ',\n')
        f.write('nb_classes :' + FLAGS.nb_classes.__str__() + ',\n')        
        f.write('data_dir :' + FLAGS.data_dir.__str__() + ',\n')
        f.write('dloss :' + dloss.__str__() + ',\n')
        f.write('gloss :' + gloss.__str__() + ',\n')
        f.write('duration :' + duration.__str__() + ',\n')

        f.close()
    except Exception as e:
        print('Unable to write ', info_file, ':', e)
        raise


def run_training():

    if FLAGS.cluster:
      with open(FLAGS.cluster) as data_file:    
        cluster_spec = json.load(data_file)  

      cluster =   tf.train.ClusterSpec(cluster_spec)

      server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

      if FLAGS.job_name == "ps":
        server.join()

    # construct the graph
    with tf.Graph().as_default():            

        size = np.array([33, 33, 33, 1])

        # read the images and labels to encode for the generator network 'fake' 
        fake_x, fake_y_ = nn.inputs(batch_size=FLAGS.batch_size,
                              num_epochs=FLAGS.num_epochs,
                              filenames=[FLAGS.generator], 
                              namescope="input_generator", 
                              size=size)

        # read the images and labels for the discriminator network 'real'
        real_x, real_y_ = nn.inputs(batch_size=FLAGS.batch_size,
                              num_epochs=FLAGS.num_epochs,
                              filenames=[FLAGS.discriminator], 
                              size=size,
                              namescope="input_discriminator")

        keep_prob = tf.placeholder(tf.float32)
        
        ps_device="/gpu:0"
        w_device="/gpu:0"
        # run the generator network on the 'fake' input images (encode/decode)
        with tf.variable_scope("generator") as scope:
          gen_x = nn.generator(fake_x, size, keep_prob, FLAGS.batch_size, ps_device=ps_device, w_device=w_device)

        with tf.variable_scope("discriminator") as scope:
          # run the discriminator network on the generated images
          gen_y_conv = nn.discriminator(gen_x, size, keep_prob, FLAGS.batch_size, ps_device=ps_device, w_device=w_device)

          scope.reuse_variables()
          # run the discriminator network on the real images
          real_y_conv = nn.discriminator(real_x, size, keep_prob, FLAGS.batch_size, ps_device=ps_device, w_device=w_device)


        # self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        # self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        # self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        # self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        # self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                              
        # self.d_loss = self.d_loss_real + self.d_loss_fake


        # calculate the loss for the real images
        loss_real_d = nn.loss(real_y_conv, real_y_)
        tf.summary.scalar("loss_real_d", loss_real_d)
        # calculate the loss for the fake images
        loss_fake_d = nn.loss(gen_y_conv, fake_y_)
        tf.summary.scalar("loss_fake_d", loss_fake_d)
        # calculate the loss for the discriminator
        loss_d = loss_real_d + loss_fake_d
        tf.summary.scalar("loss_d", loss_d)

        # calculate the loss for the generator, i.e., trick the discriminator
        loss_g = nn.loss(gen_y_conv, real_y_)
        tf.summary.scalar("loss_g", loss_g)

        vars_train = tf.trainable_variables()

        vars_gen = [var for var in vars_train if 'generator' in var.name]        
        vars_dis = [var for var in vars_train if 'discriminator' in var.name]    

        for var in vars_gen:
          print('gen', var.name)

        for var in vars_dis:
          print('dis', var.name)

        # setup the training operations        
        train_op_d = nn.training_adam(loss_d, FLAGS.learning_rate, FLAGS.beta1, FLAGS.beta2, FLAGS.epsilon, FLAGS.use_locking, "train_discriminator", vars_dis)

        train_op_g = nn.training_adam(loss_g, FLAGS.learning_rate, FLAGS.beta1, FLAGS.beta2, FLAGS.epsilon, FLAGS.use_locking, "train_generator", vars_gen)

        # caculate the accuracy
        accreal = nn.evaluation(real_y_conv, real_y_, name="accuracy_real")
        tf.summary.scalar(accreal.op.name, accreal)

        accfake = nn.evaluation(gen_y_conv, fake_y_, name="accuracy_fake")
        tf.summary.scalar(accfake.op.name, accfake)

        accuracy = (accreal + accfake)/2.0
        tf.summary.scalar("accuracy", accuracy)

        # setup the summary ops to use TensorBoard
        summary_op = tf.summary.merge_all()

        # init to setup the initial values of the weights
        #init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # create the session
        # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session() as sess:

            sess.run(init_op)
            # setup a saver for saving checkpoints
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir, sess.graph)

            # setup the coordinato and threadsr.  Used for multiple threads to read data.
            # Not strictly required since we don't have a lot of data but typically
            # using multiple threads to read data improves performance
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            start_training_time = time.time()
            # loop will continue until we run out of input training cases
            try:
                step = 0
                while not coord.should_stop():
                    # start time and run one training iteration
                    start_time = time.time()
                    
                    _g, _d, l_g, l_d, acc = sess.run([train_op_g, train_op_d, loss_g, loss_d, accuracy], feed_dict={keep_prob: 0.5})  # Update the discriminator

                    duration = time.time() - start_time

                    # print some output periodically
                    if step % 20 == 0:
                        print('OUTPUT: Step %d: loss_g = %.3f, loss_d = %3.f, accuracy = %.3f, (%.3f sec)' % (step, l_g, l_d, acc, duration))
                        # output some data to the log files for tensorboard
                        summary_str = sess.run(summary_op, feed_dict={keep_prob: 0.5})
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                    # less frequently output checkpoint files.  Used for evaluating the model
                    if step % 1000 == 0:
                        checkpoint_path = os.path.join(check_save, 'model.ckpt')
                        saver.save(sess, save_path=checkpoint_path, global_step=step)
                    step += 1

            # quit after we run out of input files to read
            except tf.errors.OutOfRangeError:
                print('OUTPUT: Done training for %d epochs, %d steps.' % (FLAGS.num_epochs,
                                                                          step))
                checkpoint_path = os.path.join(check_save, 'model.ckpt')

                saver.save(sess, checkpoint_path, global_step=step)

            finally:
                coord.request_stop()

            # shut down the threads gracefully
            coord.join(threads)
            sess.close()
            end_training_time = time.time()
            #writeTxt(dLoss, gLoss, end_training_time - start_training_time)


def main(_):
    
    info_file = os.path.join(check_save, FLAGS.txt_file)

    print(GREEN + 'learning_rate :', FLAGS.learning_rate)
    print('decay_rate :', FLAGS.decay_rate)

    print('decay_steps :', FLAGS.decay_steps)
    print('num_epochs :', FLAGS.num_epochs)
    print('batch_size :', FLAGS.batch_size)    
    print('generator :', FLAGS.generator)
    print('discriminator :', FLAGS.discriminator)
    print('checkpoint_dir :', check_save + NC)

    if not os.path.isdir(check_save):
        os.makedirs(check_save)

    run_training()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--learning_rate', 
      type=float,
      default=1e-3, 
      help='Initial learning rate.')    

    parser.add_argument(
          '--decay_rate', 
          type=float,
          default=0.96, 
          help='Learning rate decay. For gradient descent optimizer')
    parser.add_argument(
          '--decay_steps', 
          type=int,
          default=10000, 
          help='Steps at each learning rate. For gradient descent optimizer')

    parser.add_argument(
          '--beta1', 
          type=float,
          default=0.9, 
          help='For Adam optimizer')
    parser.add_argument(
          '--beta2', 
          type=float,
          default=0.999, 
          help='For Adam optimizer')
    parser.add_argument(
          '--epsilon', 
          type=float,
          default=1e-8, 
          help='For Adam optimizer')

    parser.add_argument(
          '--use_locking', 
          type=bool,
          default=False, 
          help='For Adam optimizer')

    parser.add_argument(
          '--num_epochs', 
          type=int,
          default=2, 
          help='Number of epochs to run trainer.')

    parser.add_argument(
          '--batch_size', 
          type=int,
          default=64, 
          help='Batch size.')    

    parser.add_argument(
          '--generator',
          type=str,
          required=True,
          help='tfRecords file for the generator class')

    parser.add_argument(
          '--discriminator',
          type=str,
          required=True,
          help='tfRecords file for the discriminator class')

    parser.add_argument(
          '--checkpoint_dir', 
          required=True, 
          help="Directory where to write model checkpoints.")   

    parser.add_argument(
          '--cluster', 
          type=str,
          help='JSON file with cluster specification.')     

    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )

    parser.add_argument(
          '--txt_file', 
          type=str,
          default='InfoData.txt', 
          help='Text file where to write the network information.') 



    FLAGS, unparsed = parser.parse_known_args()
    
    check_save = FLAGS.checkpoint_dir 

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
