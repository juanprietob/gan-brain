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

from datetime import datetime
import time
import os.path
import tensorflow as tf
import numpy as np
import nrrd
import neuralnetwork as nn

GREEN = "\033[0;32m"
RED = "\033[0;31m"
NC = "\033[0m"


# set the training and validation file names

TEST_FILE1 = 'test_75_85.tfrecords'
TEST_FILE2 = 'test_flipped.tfrecords'
VALIDATION_FILE = 'val_images.tfrecords'

TEST_FILES = [TEST_FILE1, TEST_FILE2]

z_dimensions = 112

# flags is a TensorFlow way to manage command line flags.

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_string('data_dir', '../Data/tfrecords/test/',
                    'Directory with the image data.')
flags.DEFINE_string('eval_dir', '../Data/2D/eval/',
                    """Directory where to write event logs.""")
flags.DEFINE_string('eval_data', 'test',
                    """Either 'train' or 'eval'""")
flags.DEFINE_string('checkpoint_dir', 'Results/checkpoints_0/',
                    """Directory where to read model checkpoints.""")
flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     """How often to run the eval.""")
flags.DEFINE_boolean('run_once', True,
                     """Whether to run eval only once.""")
flags.DEFINE_integer('nb_classes', 5, 'Number of classes.')


print(GREEN + 'batch_size :', FLAGS.batch_size)
print('data_dir :', FLAGS.data_dir)
print('eval_dir :', FLAGS.eval_dir)
print('eval_data :', FLAGS.eval_data)
print('data_dir :', FLAGS.data_dir)
print('nb_classes :', FLAGS.nb_classes)
print('checkpoint_dir :', FLAGS.checkpoint_dir + NC)


def run_eval():
    # Run evaluation on the input data set
    with tf.Graph().as_default() as g:

        # Get images and labels for the MRI data
        eval_data = FLAGS.eval_data == 'eval'

        # specify the training data file location
        testfiles = []

        for fi in TEST_FILES:
            testfiles.append(os.path.join(FLAGS.data_dir, fi))

        # read the proper data set
        images, labels = nn.inputs(batch_size=FLAGS.batch_size,
                                   num_epochs=1, filenames=testfiles, ifeval=True)

        # Build a Graph that computes the logits predictions from the
        # inference model.  We'll use a prior graph built by the training
        z_placeholder = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dimensions])

        _, Gz, _ = nn.inference(images, z_placeholder, z_dimensions, FLAGS.batch_size, 'yo.txt', False)


        # Calculate predictions.
        # pred, lab, acc = nn.evaluation(logits, labels)

        # setup the initialization of variables
        local_init = tf.initialize_local_variables()

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

        # create the saver and session
        saver = tf.train.Saver()
        sess = tf.Session()

        # init the local variables
        sess.run(local_init)

        while True:

            # read in the most recent checkpointed graph and weights
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found in %s' % FLAGS.checkpoint_dir)
                return

            # start up the threads
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:

                step = 0
                while not coord.should_stop():
                    # run a single iteration of evaluation
                    # print('OUTPUT: Step %d:' % step)
                    z_batch = np.random.normal(-1, 1, size=[1, z_dimensions])
                    gz = sess.run([Gz], feed_dict={z_placeholder:z_batch})

                    try:
                        data_save = FLAGS.checkpoint_dir + 'Data'
                        if not os.path.isdir(data_save):
                            os.makedirs(data_save)
                        print(np.shape(gz[0][0][:, :, 0]))
                        im = gz[0][0][:, :, 0]
                        nrrd.write(os.path.join(data_save, 'Gz_' + step.__str__() + '.nrrd'), np.reshape(im,(195, 233)))
                    except Exception as e:
                        print('Unable to save data to', 'test.npy', ':', e)
                        raise

                    step += 1

            except tf.errors.OutOfRangeError:

                print('OUTPUT: %s: ' % (datetime.now()))

                print('OUTPUT: %d images evaluated from file %s & %s' % (step, testfiles[0], testfiles[1]))

                # create summary to show in TensorBoard
                #summary = tf.Summary()
                summary = sess.run(summary_op)
                #summary.ParseFromString(sess.run(summary_op))

                summary_writer.add_summary(summary, global_step)

            finally:
                coord.request_stop()

            # shutdown gracefully
            coord.join(threads)

            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)
            sess.close()


def main(_):
    run_eval()


if __name__ == '__main__':
    tf.app.run()
