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
    for i in range(-size, size):
        for j in range(-size, size):
            for k in range(-size, size):
                neigh[i][j][k] = img[i + index[0]][j + index[1]][k + index[2]]
    return neigh


def run_generation():

    print("Reading:", FLAGS.img)
    img, head = nrrd.read(FLAGS.img)
    img = img.astype(float)

    imgmask = None
    if FLAGS.mask is not None:
        print("Reading:", FLAGS.mask)
        imgmask, headmask = nrrd.read(FLAGS.mask)
    
    imgsize = head["sizes"]
    splitimage = []
    splitimage_index = []

    for i in range(16, imgsize[0] - 16, 32):
        for j in range(16, imgsize[1] - 16, 32):
            for k in range(16, imgsize[2] - 16, 32):
                extract = True
                if imgmask is not None:
                    if imgmask[i][j][k] != 0:
                        extract = True
                    else:
                        extract = False

                if extract:
                    splitimage_index.append([i, j, k])
                    neigh = get_neighborhood(img, [i, j, k], 16)
                    splitimage.append(neigh)

    splitimage_index = np.array(splitimage_index)
    splitimage = np.array(splitimage).astype(float)
    splitimage = splitimage.reshape([splitimage.shape[0], splitimage.shape[1], splitimage.shape[2], splitimage.shape[3], 1])
    batch_size = splitimage.shape[0]

    # construct the graph
    with tf.Graph().as_default():            

        size = np.array([33, 33, 33, 1])

        # read the images and labels to encode for the generator network 'fake' 
        fake_x = tf.placeholder(tf.float32, shape=[batch_size, size[0], size[1], size[2], size[3]])

        keep_prob = 1
        
        ps_device="/gpu:0"
        w_device="/gpu:0"
        # run the generator network on the 'fake' input images (encode/decode)
        with tf.variable_scope("generator") as scope:
          gen_x = nn.generator(fake_x, size, keep_prob, batch_size, ps_device=ps_device, w_device=w_device)

        # init to setup the initial values of the weights
        #init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # create the session
        with tf.Session() as sess:

            sess.run(init_op)
            # setup a saver for saving checkpoints
            # setup the coordinato and threadsr.  Used for multiple threads to read data.
            # Not strictly required since we don't have a lot of data but typically
            # using multiple threads to read data improves performance
            start_training_time = time.time()

            generated = sess.run([gen_x], feed_dict={fake_x: splitimage})  # Update the discriminator
            # shut down the threads gracefully
            sess.close()
            end_training_time = time.time()

            generated = np.array(generated).reshape(splitimage.shape)
            
            for index, neigh in zip(splitimage_index, generated):
                for i in range(-16, 16):
                    for j in range(-16, 16):
                        for k in range(-16, 16):
                            img[i + index[0]][j + index[1]][k + index[2]] = neigh[i + 16][j + 16][k + 16]
            
            print("Writing:", FLAGS.out)
            nrrd.write(FLAGS.out, img, head)
            
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
    parser.add_argument('--out', help='Output image', default="out.nrrd", type=str)

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
