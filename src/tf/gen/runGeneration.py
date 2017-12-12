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

import nrrd
import json

GREEN = "\033[0;32m"
NC = "\033[0m"

FLAGS = None
check_save = None
cluster_spec = None

z_dimensions = 112

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = 178):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = chr(fill) * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

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

def get_neighborhood(img, index, size):

    neigh = np.zeros([size*2 + 1, size*2 + 1, size*2 + 1])

    for i, ii in enumerate(range(index[0] - size, index[0] + size + 1)):
        for j, jj in enumerate(range(index[1] - size, index[1] + size + 1)):
            for k, kk in enumerate(range(index[2] - size, index[2] + size + 1)):
                neigh[i][j][k] = img[ii][jj][kk]

    return neigh

def set_neighborhood(img, index, size, neigh, padding=0):
    
    for i, ii in enumerate(range(index[0] - size + padding, index[0] + size + 1 - padding), padding):
        for j, jj in enumerate(range(index[1] - size + padding, index[1] + size + 1 - padding), padding):
            for k, kk in enumerate(range(index[2] - size + padding, index[2] + size + 1 - padding), padding):
                img[ii][jj][kk] = neigh[i][j][k]


def run_generation():

    print("Reading:", FLAGS.img)
    img, head = nrrd.read(FLAGS.img)
    img = img.astype(float)

    imgmask = None
    if FLAGS.mask is not None:
        print("Reading:", FLAGS.mask)
        imgmask, headmask = nrrd.read(FLAGS.mask)
    
    imgsize = head["sizes"]
    neighborhood_size = 32

    img_out = np.zeros(img.shape)

    batch_size = 1

    ps_device = FLAGS.ps_device
    w_device = FLAGS.w_device

    size = np.concatenate(([1], imgsize, [1]))
    # size = np.concatenate(([1], [neighborhood_size*2 + 1, neighborhood_size*2 + 1, neighborhood_size*2 + 1], [1]))
    # padding = 4

    # construct the graph
    with tf.Graph().as_default():

        # read the images and labels to encode for the generator network 'fake' 
        fake_x = tf.placeholder(tf.float32, shape=size)

        keep_prob = 1
        
        # run the generator network on the 'fake' input images (encode/decode)
        with tf.variable_scope("generator") as scope:
            if(FLAGS.dim == 3):
                gen_x = nn.generator(fake_x, keep_prob, batch_size, ps_device=ps_device, w_device=w_device, is_training=False)
            else:
                gen_x = nn.generator2d(fake_x, keep_prob, batch_size, ps_device=ps_device, w_device=w_device, is_training=False)

        # init to setup the initial values of the weights
        #init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # create the session
        with tf.Session() as sess:

            sess.run(init_op)
            #sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            print("Restoring model:", FLAGS.model)
            saver.restore(sess, FLAGS.model)
            # setup a saver for saving checkpoints
            # setup the coordinato and threadsr.  Used for multiple threads to read data.
            # Not strictly required since we don't have a lot of data but typically
            # using multiple threads to read data improves performance
            start_training_time = time.time()

            print("I am self aware...")

            # for i in range(neighborhood_size, imgsize[0] - neighborhood_size + 1, (neighborhood_size)):
            #     printProgressBar(i - neighborhood_size + 1, imgsize[0] - neighborhood_size + 1, prefix = 'Skynet:', suffix = 'Complete', length = 50)
            #     for j in range(neighborhood_size, imgsize[1] - neighborhood_size + 1, (neighborhood_size)):
            #         for k in range(neighborhood_size, imgsize[2] - neighborhood_size + 1, (neighborhood_size)):
            #             extract = True
            #             if imgmask is not None:
            #                 if imgmask[i][j][k] != 0:
            #                     extract = True
            #                 else:
            #                     extract = False

            #             if extract:
                            
            #                 neigh = get_neighborhood(img, [i, j, k], neighborhood_size)
            #                 f_x = neigh.reshape((1,) + neigh.shape + (1,))
            #                 generated = sess.run([gen_x], feed_dict={fake_x: f_x})  # Update the discriminator
            #                 generated = np.array(generated).reshape(neigh.shape)
            #                 set_neighborhood(img_out, [i, j, k], neighborhood_size, generated, padding)

            img = img.reshape(size)
            generated = sess.run([gen_x], feed_dict={fake_x: img})
            img_out = np.array(generated).reshape(imgsize)

            # shut down the threads gracefully
            sess.close()
            end_training_time = time.time()
            
            # for index, neigh in zip(splitimage_index, generated):
            #     for i in range(-neighborhood_size, neighborhood_size):
            #         for j in range(-neighborhood_size, neighborhood_size):
            #             for k in range(-neighborhood_size, neighborhood_size):
            #                 img[i + index[0]][j + index[1]][k + index[2]] = neigh[i + neighborhood_size][j + neighborhood_size][k + neighborhood_size]
            
            print("jk, writing image:", FLAGS.out)
            nrrd.write(FLAGS.out, img_out, head)
            
            #writeTxt(dLoss, gLoss, end_training_time - start_training_time)


def main(_):

    print(GREEN) 
    print('Model :', FLAGS.model)
    print('Input image:', FLAGS.img)
    print('Output:', FLAGS.out)
    print(NC)

    run_generation()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help='Model file computed with runTraining.py', required=True)
    parser.add_argument('--img', help='Input image to generate', required=True)
    parser.add_argument('--mask', help='Input image mask (work on this area only)')
    parser.add_argument('--dim', help='Image dimensionality', default=3, type=int)
    parser.add_argument('--out', help='Output image', default="out.nrrd", type=str)
    parser.add_argument('--ps_device', help='Process device, to store memory', default="/gpu:0", type=str)
    parser.add_argument('--w_device', help='Process device, to store memory', default="/gpu:0", type=str)
    

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
