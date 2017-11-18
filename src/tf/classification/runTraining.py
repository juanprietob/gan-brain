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

tf.logging.set_verbosity(tf.logging.DEBUG)

GREEN = "\033[0;32m"
NC = "\033[0m"

TRAIN_FILES = ['train.tfRecords']

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-9, 'Initial learning rate.')
flags.DEFINE_float('decay_rate', 0.96, 'Learning rate decay.')
flags.DEFINE_integer('decay_steps', 100000, 'Steps at each learning rate.')
flags.DEFINE_integer('num_epochs', 200, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('nb_classes', 2, 'Number of classes.')
flags.DEFINE_string('data_dir', '../../data/tfRecords/', 'Directory with the training data.')
flags.DEFINE_string('checkpoint_dir', '../../data/Results/checkpoints', "Directory where to write model checkpoints.")
flags.DEFINE_string('txt_file', 'InfoData.txt', 'Text file where to write the network information.')

z_dimensions = 112

check_save = FLAGS.checkpoint_dir + '_' + '/'
info_file = os.path.join(check_save, FLAGS.txt_file)

print(GREEN + 'learning_rate :', FLAGS.learning_rate)
print('decay_rate :', FLAGS.decay_rate)
print('decay_steps :', FLAGS.decay_steps)
print('num_epochs :', FLAGS.num_epochs)
print('batch_size :', FLAGS.batch_size)
print('nb_classes :', FLAGS.nb_classes)
print('data_dir :', FLAGS.data_dir)
print('checkpoint_dir :', check_save + NC)


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


if not os.path.isdir(check_save):
    os.makedirs(check_save)


def run_training():
    # construct the graph
    with tf.Graph().as_default():

        # specify the training data file location
        trainfiles = []

        for fi in TRAIN_FILES:
            trainfiles.append(os.path.join(FLAGS.data_dir, fi))

            # trainfile = os.path.join(FLAGS.data_dir, TRAIN_FILE)

        # read the images and labels
        x, y_ = nn.inputs(batch_size=FLAGS.batch_size,
                              num_epochs=FLAGS.num_epochs,
                              filenames=trainfiles, ifeval=False)
        keep_prob = tf.placeholder(tf.float32)
        
        z_placeholder = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dimensions])

        # run inference on the images
        y_conv = nn.inference(x, np.array([65, 65, 65]), keep_prob, FLAGS.batch_size)

        # calculate the loss from the results of inference and the labels
        loss = nn.loss(y_conv, y_)   

        # caculate the accuracy
        accuracy = nn.evaluation(y_conv, y_)     

        # setup the training operations        
        train_op = nn.training(loss, FLAGS.learning_rate, FLAGS.decay_steps, FLAGS.decay_rate)

        # setup the summary ops to use TensorBoard
        summary_op = tf.summary.merge_all()

        # init to setup the initial values of the weights
        #init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # create the session
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
                    
                    _, l, acc = sess.run([train_op, loss, accuracy], feed_dict={keep_prob: 0.5})  # Update the discriminator

                    duration = time.time() - start_time

                    # print some output periodically
                    if step % 20 == 0:
                        print('OUTPUT: Step %d: loss = %.3f (%.3f sec), accuracy = %.3f' % (step, l, duration, acc))
                        # output some data to the log files for tensorboard
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                    # less frequently output checkpoint files.  Used for evaluating the model
                    if step % 500 == 0:
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
    run_training()


if __name__ == '__main__':
    tf.app.run()
