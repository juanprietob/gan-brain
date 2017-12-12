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

from datetime import datetime

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

        size = np.array(FLAGS.img_size)

        # read the images and labels to encode for the generator network 'fake' 
        _x, _y_ = nn.inputs(batch_size=FLAGS.batch_size,
                              num_epochs=FLAGS.num_epochs,
                              filenames=[FLAGS.tf_records], 
                              size=size,
                              namescope="input_generator",)

        keep_prob = tf.placeholder(tf.float32)
        
        ps_device="/gpu:0"
        w_device="/gpu:0"
        # run the generator network on the 'fake' input images (encode/decode)
        with tf.variable_scope("generator") as scope:
          if(len(size) == 4):
            gen_x = nn.generator(_x, keep_prob, FLAGS.batch_size, ps_device=ps_device, w_device=w_device, is_training=True)
          else:
            gen_x = nn.generator2d(_x, keep_prob, FLAGS.batch_size, ps_device=ps_device, w_device=w_device, is_training=True)

        _y_ = tf.layers.batch_normalization(_y_)

        # calculate the loss for the generator, i.e., trick the discriminator
        loss_g = nn.loss(gen_x, _y_)
        tf.summary.scalar("loss_g", loss_g)
        
        # setup the training operations        
        train_op_g = nn.training_adam(loss_g, FLAGS.learning_rate, FLAGS.beta1, FLAGS.beta2, FLAGS.epsilon, FLAGS.use_locking, "train_discriminator")

        # calculate the accuracy

        accuracy = nn.evaluation(gen_x, _y_, name="accuracy")

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
            now = datetime.now()
            summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name + "-" + now.strftime("%Y%m%d-%H%M%S")), sess.graph)

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
                    
                    _g, l_g, acc = sess.run([train_op_g, loss_g, accuracy], feed_dict={keep_prob: 0.5})  # Update the discriminator

                    duration = time.time() - start_time

                    # print some output periodically
                    if step % 20 == 0:
                      print('OUTPUT: Step', step, 'loss:', l_g, 'accuracy:', acc, 'duraction:', duration)
                      #print('OUTPUT: Step %d: loss_g = %.3f, accuracy = %.3f, (%.3f sec)' % (step, l_g, acc, duration))
                      # output some data to the log files for tensorboard
                      summary_str = sess.run(summary_op, feed_dict={keep_prob: 0.5})
                      summary_writer.add_summary(summary_str, step)
                      summary_writer.flush()

                    # less frequently output checkpoint files.  Used for evaluating the model
                    if step % 1000 == 0:
                        checkpoint_path = os.path.join(check_save, FLAGS.model_name)
                        saver.save(sess, save_path=checkpoint_path, global_step=step)
                        print('MODEL:', checkpoint_path)
                    step += 1

            # quit after we run out of input files to read
            except tf.errors.OutOfRangeError:
                print('OUTPUT: Done training for %d epochs, %d steps.' % (FLAGS.num_epochs,
                                                                          step))
                checkpoint_path = os.path.join(check_save, FLAGS.model_name)

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
    print('tf_records :', FLAGS.tf_records)
    print('img_size :', FLAGS.img_size)
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
          '--tf_records',
          type=str,
          required=True,
          help='tfRecords file to train a generation network, generate input image into label')

    parser.add_argument(
          '--img_size',
          nargs='+',
          type=int,
          default=[33,33,33,1],
          help='Image size in the tfRecords width,height,depth,num_channels ')

    parser.add_argument(
          '--model_name',
          type=str,
          default="model",
          help='tfRecords file to train a generation network, generate input image into label')

    # parser.add_argument(
    #       '--discriminator',
    #       type=str,
    #       required=True,
    #       help='tfRecords file for the discriminator class')

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
