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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time

import sys
sys.path.append('../util')

from netFunctions import print_tensor_shape


def read_and_decode(filename_queue):
    # input: filename
    # output: image, label pair

    # setup a TF record reader
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # list the features we want to extract, i.e., the image and the label
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })

    # Set image and label shapes
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)

    # Decode the training image
    image = tf.decode_raw(features['raw'], tf.float32)    
    image_re = tf.reshape(image, ([65, 65, 65, 1]))
    print_tensor_shape(image_re, 'Training image')

    # Decode label

    label = tf.cast(features['label'], tf.int32)        

    return image_re, label


def inputs(batch_size, num_epochs, filenames, ifeval):
    # inputs: batch_size, num_epochs are scalars, filename
    # output: image and label pairs for use in training or eval

    if not num_epochs: num_epochs = None

    if ifeval: num_images_per_epoch = 1000
    else: num_images_per_epoch = 20000

    min_queue_examples = int(num_images_per_epoch * 0.8)

    # define the input node
    with tf.name_scope('input'):
        # setup a TF filename_queue
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs)

        # return and image and label
        image, label = read_and_decode(filename_queue)

        # shuffle the images, not strictly necessary as the data creating
        # phase already did it, but there's no harm doing it again.
        images, labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=4,
            capacity=50000,
            min_after_dequeue=10000)

        labels = tf.one_hot(labels, 2)
        print_tensor_shape(labels, 'Training labels')


        beta = tf.Variable(tf.constant(0.0, shape=[1]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[1]),
                                      name='gamma', trainable=True)
        mean, variance = tf.nn.moments(images, [0])
        images = tf.nn.batch_normalization(images, mean, variance, beta, gamma, 1e-3)

        return images, labels


def inference(images, size, keep_prob=1, batch_size=1, regularization_constant=0.0):

#   input: tensor of images
#   output: tensor of computed logits

# resize the image tensors to add the number of channels, 1 in this case
# required to pass the images to various layers upcoming in the graph
    #print("Image size:", size)
    #num_channels = size[0], depth = size[0], height = size[1], width = size[2], num_channels = size[3]
    print_tensor_shape(images, "images")
# Convolution layer
    with tf.name_scope('Conv1'):

# weight variable 4d tensor, first two dims are patch (kernel) size       
# third dim is number of input channels and fourth dim is output channels
        W_conv1 = tf.Variable(tf.truncated_normal([3,3,3,1,64],stddev=0.1,
                     dtype=tf.float32),name='W_conv1')
        print_tensor_shape( W_conv1, 'W_conv1 shape')

        conv1_op = tf.nn.conv3d( images, W_conv1, strides=[1,2,2,2,1], 
                     padding="SAME", name='conv1_op' )
        print_tensor_shape( conv1_op, 'conv1_op shape')

        W_bias1 = tf.Variable( tf.zeros([1,33,33,33,64], dtype=tf.float32), 
                          name='W_bias1')
        print_tensor_shape( W_bias1, 'W_bias1 shape')

        bias1_op = conv1_op + W_bias1
        print_tensor_shape( bias1_op, 'bias1_op shape')

        relu1_op = tf.nn.relu( bias1_op, name='relu1_op' )
        print_tensor_shape( relu1_op, 'relu1_op shape')

# Pooling layer
    # with tf.name_scope('Pool1'):
    #     pool1_op = tf.nn.max_pool3d(relu1_op, ksize=[1,2,2,2,1],
    #                               strides=[1,2,2,2,1], padding='SAME') 
    #     print_tensor_shape( pool1_op, 'pool1_op shape')

# Conv layer
    with tf.name_scope('Conv2'):
        W_conv2 = tf.Variable(tf.truncated_normal([3,3,3,64,128],stddev=0.1,
                     dtype=tf.float32),name='W_conv2')
        print_tensor_shape( W_conv2, 'W_conv2 shape')

        conv2_op = tf.nn.conv3d( relu1_op, W_conv2, strides=[1,2,2,2,1],
                     padding="SAME", name='conv2_op' )
        print_tensor_shape( conv2_op, 'conv2_op shape')

        W_bias2 = tf.Variable( tf.zeros([1,17,17,17,128], dtype=tf.float32),
                          name='W_bias2')
        print_tensor_shape( W_bias2, 'W_bias2 shape')

        bias2_op = conv2_op + W_bias2
        print_tensor_shape( bias2_op, 'bias2_op shape')

        relu2_op = tf.nn.relu( bias2_op, name='relu2_op' )
        print_tensor_shape( relu2_op, 'relu2_op shape')

# Pooling layer
    # with tf.name_scope('Pool2'):
    #     pool2_op = tf.nn.max_pool3d(relu2_op, ksize=[1,2,2,2,1],
    #                               strides=[1,2,2,2,1], padding='SAME')
    #     print_tensor_shape( pool2_op, 'pool2_op shape')
    
# Conv layer
    with tf.name_scope('Conv3'):
        W_conv3 = tf.Variable(tf.truncated_normal([3,3,3,128,256],stddev=0.1,
                     dtype=tf.float32),name='W_conv3') 
        print_tensor_shape( W_conv3, 'W_conv3 shape')

        conv3_op = tf.nn.conv3d( relu2_op, W_conv3, strides=[1,2,2,2,1],
                     padding='SAME', name='conv3_op' )
        print_tensor_shape( conv3_op, 'conv3_op shape')

        W_bias3 = tf.Variable( tf.zeros([1,9,9,9,256], dtype=tf.float32),
                          name='W_bias3')
        print_tensor_shape( W_bias3, 'W_bias3 shape')

        bias3_op = conv3_op + W_bias3
        print_tensor_shape( bias3_op, 'bias3_op shape')

        relu3_op = tf.nn.relu( bias3_op, name='relu3_op' )
        print_tensor_shape( relu3_op, 'relu3_op shape')
    
# Conv layer
    with tf.name_scope('Conv4'):
        W_conv4 = tf.Variable(tf.truncated_normal([3,3,3,256,512],stddev=0.1,
                    dtype=tf.float32), name='W_conv4')
        print_tensor_shape( W_conv4, 'W_conv4 shape')

        conv4_op = tf.nn.conv3d( relu3_op, W_conv4, strides=[1,2,2,2,1],
                     padding='SAME', name='conv4_op' )
        print_tensor_shape( conv4_op, 'conv4_op shape')

        W_bias4 = tf.Variable( tf.zeros([1,5,5,5,512], dtype=tf.float32),
                          name='W_bias4')
        print_tensor_shape( W_bias4, 'W_bias4 shape')

        bias4_op = conv4_op + W_bias4
        print_tensor_shape( bias4_op, 'bias4_op shape')

        relu4_op = tf.nn.relu( bias4_op, name='relu4_op' )
        print_tensor_shape( relu4_op, 'relu4_op shape')

        drop_op = tf.nn.dropout( relu4_op, keep_prob )
        print_tensor_shape( drop_op, 'drop_op shape' )

#Fully connected layer
    with tf.name_scope('Matmul5'):

        W_matmul5 = tf.Variable(tf.truncated_normal([5*5*5*512,512],stddev=0.1,
                    dtype=tf.float32), name='W_matmul5')
        print_tensor_shape( W_matmul5, 'W_matmul5 shape')        

        W_bias5 = tf.Variable( tf.zeros([512], dtype=tf.float32),
                          name='W_bias5')

        shape = drop_op.get_shape().as_list()
        
        h_drop_op_flat = tf.reshape(drop_op, [-1, shape[1]*shape[2]*shape[3]*shape[4]])
        matmul5_op = tf.nn.relu(tf.matmul(h_drop_op_flat, W_matmul5) + W_bias5)

    with tf.name_scope('Matmul6'):

        W_matmul6 = tf.Variable(tf.truncated_normal([512,2],stddev=0.1,
                    dtype=tf.float32), name='W_matmul6')
        print_tensor_shape( W_matmul6, 'W_matmul6 shape')        

        W_bias6 = tf.Variable( tf.zeros([2], dtype=tf.float32),
                          name='W_bias6')

        matmul6_op = tf.matmul(matmul5_op, W_matmul6) + W_bias6
        

#Conv layer to generate the 2 score classes
    # with tf.name_scope('Score_classes'):
    #     W_score_classes = tf.Variable(tf.truncated_normal([1,1,1,1024,2],
    #                         stddev=0.1,dtype=tf.float32),name='W_score_classes')
    #     print_tensor_shape( W_score_classes, 'W_score_classes_shape')

    #     score_classes_conv_op = tf.nn.conv3d( drop_op, W_score_classes, 
    #                    strides=[1,1,1,1,1], padding='SAME', 
    #                    name='score_classes_conv_op')
    #     print_tensor_shape( score_classes_conv_op,'score_conv_op shape')

    #     W_bias5 = tf.Variable( tf.zeros([1,3,3,11,2], dtype=tf.float32),
    #                       name='W_bias5')
    #     print_tensor_shape( W_bias5, 'W_bias5 shape')

    #     bias5_op = score_classes_conv_op + W_bias5
    #     print_tensor_shape( bias5_op, 'bias5_op shape')

# Upscore the results to 256x256x2 image
    # with tf.name_scope('Upscore1'):
    #     W_upscore1 = tf.Variable(tf.truncated_normal([8,8,1,2,2],
    #                           stddev=0.1,dtype=tf.float32),name='W_upscore1')
    #     print_tensor_shape( W_upscore1, 'W_upscore1 shape')
      
    #     upscore1_conv_op = tf.nn.conv3d_transpose( bias5_op, 
    #                    W_upscore1,
    #                    output_shape=[batch_size,11,11,11,2],strides=[1,4,4,1,1],
    #                    padding='SAME',name='upscore1_conv_op')
    #     print_tensor_shape(upscore1_conv_op, 'upscore1_conv_op shape')

    #Regularization of all the weights in the network for the loss function
    # with tf.name_scope('Regularization'):
    #   Reg_constant = tf.constant(regularization_constant)
    #   reg_op = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv4) + tf.nn.l2_loss(W_score_classes) #+ tf.nn.l2_loss(W_conv3)
    #   reg_op = reg_op*Reg_constant
    #   tf.summary.scalar('reg_op', reg_op)

    return matmul6_op


def loss(logits, labels):

    print_tensor_shape( logits, 'logits shape')
    print_tensor_shape( labels, 'labels shape')

    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy')

    loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

    return loss


def training(loss, learning_rate, decay_steps, decay_rate):
    # input: loss: loss tensor from loss()
    # input: learning_rate: scalar for gradient descent
    # output: train_op the operation for training

#    Creates a summarizer to track the loss over time in TensorBoard.

#    Creates an optimizer and applies the gradients to all trainable variables.

#    The Op returned by this function is what must be passed to the
#    `sess.run()` call to cause the model to train.

  # Add a scalar summary for the snapshot loss.

  # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

  # create learning_decay
    lr = tf.train.exponential_decay( learning_rate,
                                     global_step,
                                     decay_steps,
                                     decay_rate, staircase=True )

    tf.summary.scalar('2learning_rate', lr )

  # Create the gradient descent optimizer with the given learning rate.
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(lr)

  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def evaluation(logits, labels):
    with tf.name_scope('Accuracy'):
        values, indices = tf.nn.top_k(labels, 1);
        correct = tf.reshape(tf.nn.in_top_k(logits, tf.cast(tf.reshape( indices, [-1 ] ), tf.int32), 1), [-1] )
        print_tensor_shape( correct, 'correct shape')
        return tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

# def evaluation(logits, labels):
#     # input: logits: Logits tensor, float - [batch_size, 195, 233, NUM_CLASSES].
#     # input: labels: Labels tensor, int32 - [batch_size, 195, 233]
#     # output: scaler int32 tensor with number of examples that were 
#     #         predicted correctly

#     with tf.name_scope('eval'):
#         print()
#         print_tensor_shape(logits, 'logits eval shape before')
#         print_tensor_shape(labels, 'labels eval shape before')

#         # reshape to match args required for the top_k function
#         logits_re = tf.reshape(logits, [-1])
#         print_tensor_shape(logits_re, 'logits_re eval shape after')
#         labels_re = tf.reshape(labels, [-1])
#         print_tensor_shape(labels_re, 'labels_re eval shape after')

#         # get accuracy :
#         diff = tf.sub(labels_re,logits_re)
#         acc = tf.div(tf.reduce_mean(diff), 195.0*233.0)
#         acc = 1 - acc

#         # get accuracy :
#         diff = tf.abs(tf.sub(labels_re,logits_re))
#         lessthan0_01 = tf.less_equal(diff, 0.01)
#         sum = tf.reduce_sum(tf.cast(lessthan0_01, tf.float32))
#         acc2 = tf.div(sum, 195.0*233.0)

#         print(acc)

#         # Return the tuple of intersection, label and example areas
#         labels_re = tf.cast(labels_re, tf.float32)
#         indices_re = tf.cast(logits_re, tf.float32)
#         return indices_re, labels_re, acc2
