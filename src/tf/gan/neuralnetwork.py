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

import numpy as np

import sys
sys.path.append('../util')

from netFunctions import print_tensor_shape


def read_and_decode(filename_queue, size, namescope):
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
    image_re = tf.reshape(image, (size))
    print_tensor_shape(image_re, namescope + ' image')

    # Decode label

    label = tf.cast(features['label'], tf.int32)        

    return image_re, label


def inputs(batch_size, num_epochs, filenames, size, namescope="input"):
    # inputs: batch_size, num_epochs are scalars, filename
    # output: image and label pairs for use in training or eval    

    # define the input node
    with tf.name_scope(namescope):
        # setup a TF filename_queue
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs)

        # return and image and label
        image, label = read_and_decode(filename_queue, size, namescope)

        # shuffle the images, not strictly necessary as the data creating
        # phase already did it, but there's no harm doing it again.
        images, labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=4,
            capacity=50000,
            min_after_dequeue=10000)

        labels = tf.one_hot(labels, 2)
        print_tensor_shape(labels, namescope + ' labels')


        beta = tf.Variable(tf.constant(0.0, shape=[1]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[1]),
                                      name='gamma', trainable=True)
        mean, variance = tf.nn.moments(images, [0])
        images = tf.nn.batch_normalization(images, mean, variance, beta, gamma, 1e-3)

        return images, labels

def convolution(images, out_channels, name, relu=True, ps_device="/cpu:0", w_device="/gpu:0"):

    in_channels = images.get_shape()[-1]

    with tf.variable_scope(name):
# weight variable 4d tensor, first two dims are patch (kernel) size       
# third dim is number of input channels and fourth dim is output channels
        with tf.device(ps_device):

            w_conv_name = 'w_' + name
            w_conv = tf.get_variable(w_conv_name, shape=[3,3,3,in_channels,out_channels], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
            print_tensor_shape( w_conv, 'weight shape')

            b_conv_name = 'b_' + name
            b_conv = tf.get_variable(b_conv_name, shape=[out_channels])
            print_tensor_shape( b_conv, 'bias shape')

        with tf.device(w_device):
            conv_op = tf.nn.conv3d( images, w_conv, strides=[1,2,2,2,1], padding="SAME", name='conv1_op' )
            print_tensor_shape( conv_op, 'conv_op shape')

            conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

            if(relu):
                conv_op = tf.nn.relu( conv_op, name='relu_op' ) 
                print_tensor_shape( conv_op, 'relu_op shape')

            return conv_op

def deconvolution(images, output_shape, name, relu=True, ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):

        in_channels = images.get_shape()[-1]

        out_channels = output_shape[4]

        with tf.device(ps_device):
            w_deconv_name = 'w_' + name
            w_deconv = tf.get_variable(w_deconv_name, shape=[3,3,3,out_channels,in_channels], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
            print_tensor_shape( w_deconv, 'weight shape')

            b_deconv_name = 'b_' + name
            b_deconv = tf.get_variable(b_deconv_name, shape=[out_channels])
            print_tensor_shape( b_deconv, 'bias shape')

        with tf.device(w_device):

            deconv_op = tf.nn.conv3d_transpose( images, w_deconv, 
                output_shape=output_shape,
                strides=[1,2,2,2,1],
                padding='SAME', name='deconv_op' )

            print_tensor_shape( w_deconv, 'deconv_op shape')

            deconv_op = tf.nn.bias_add(deconv_op, b_deconv, name='bias_add_op')

        if relu:
            deconv_op = tf.nn.relu( deconv_op, name='relu_op' )
            print_tensor_shape( deconv_op, 'relu_op shape')

        return deconv_op

def matmul(images, out_channels, name, relu=True, ps_device="/cpu:0", w_device="/gpu:0"):

     with tf.variable_scope(name):

        shape = images.get_shape().as_list()

        with tf.device(ps_device):
            w_matmul_name = 'w_' + name
            w_matmul = tf.get_variable(w_matmul_name, shape=[shape[1],out_channels], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
        
            print_tensor_shape( w_matmul, 'w_matmul shape')        

            b_matmul_name = 'b_' + name
            b_matmul = tf.get_variable(name='b_matmul_name', shape=[out_channels])        

        with tf.device(w_device):

            matmul_op = tf.nn.bias_add(tf.matmul(images, w_matmul), b_matmul)

            if(relu):
                matmul_op = tf.nn.relu(matmul_op)

            return matmul_op


def generator(images, size, keep_prob=1, batch_size=1, regularization_constant=0.0, ps_device="/cpu:0", w_device="/gpu:0"):

# Encoder part of the network

#   input: tensor of images
#   output: tensor of computed logits

# resize the image tensors to add the number of channels, 1 in this case
# required to pass the images to various layers upcoming in the graph
    #print("Image size:", size)
    #num_channels = size[0], depth = size[0], height = size[1], width = size[2], num_channels = size[3]
    print_tensor_shape(images, "images")
# Convolution layer
    # with tf.variable_scope('Conv1'):

    #     with tf.device("/cpu:0"):

    #         W_conv1 = tf.get_variable("W_conv1", shape=[3,3,3,1,64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
    #         print_tensor_shape( W_conv1, 'W_conv1 shape')

    #         W_bias1 = tf.get_variable('W_bias1', shape=[1,17,17,17,64])
    #         print_tensor_shape( W_bias1, 'W_bias1 shape')

    #     with tf.device("/gpu:0"):
    #         conv1_op = tf.nn.conv3d( images, W_conv1, strides=[1,2,2,2,1], padding="SAME", name='conv1_op' )
    #         print_tensor_shape( conv1_op, 'conv1_op shape')

    #         bias1_op = conv1_op + W_bias1
    #         print_tensor_shape( bias1_op, 'bias1_op shape')

    #         relu1_op = tf.nn.relu( bias1_op, name='relu1_op' )
    #         print_tensor_shape( relu1_op, 'relu1_op shape')

    relu1_op = convolution(images, 32, "Conv1", ps_device=ps_device, w_device=w_device)

    relu2_op = convolution(relu1_op, 256, "Conv2", ps_device=ps_device, w_device=w_device)

    relu3_op = convolution(relu2_op, 1024, "Conv3", ps_device=ps_device, w_device=w_device)

    relu4_op = convolution(relu3_op, 4048, "Conv4", ps_device=ps_device, w_device=w_device)

    with tf.device(w_device):
        drop_op = tf.nn.dropout( relu4_op, keep_prob )
        print_tensor_shape( drop_op, 'drop_op shape' )

    #     shape =  drop_op.get_shape().as_list()
    #     h_drop_flat = tf.reshape(drop_op, [-1, shape[1]*shape[2]*shape[3]*shape[4]])

    # matmul5_op = matmul(h_drop_flat, 16384, "Matmul5", ps_device=ps_device, w_device=w_device)

    # matmul6_flat = matmul(matmul5_op, shape[1]*shape[2]*shape[3]*shape[4], "Matmul6", ps_device=ps_device, w_device=w_device)

    # with tf.device(w_device):

    #     matmul6_op = tf.reshape(matmul6_flat, [-1, shape[1], shape[2], shape[3], shape[4]])    
        
    shape = relu3_op.get_shape().as_list()
    deconv1_op = deconvolution(drop_op, [batch_size,shape[1],shape[2],shape[3],1024], "Deconv1", ps_device=ps_device, w_device=w_device)

    shape = relu2_op.get_shape().as_list()
    deconv2_op = deconvolution(deconv1_op, [batch_size,shape[1],shape[2],shape[3],256], "Deconv2", ps_device=ps_device, w_device=w_device)

    shape = relu1_op.get_shape().as_list()
    deconv3_op = deconvolution(deconv2_op, [batch_size,shape[1],shape[2],shape[3],32], "Deconv3", ps_device=ps_device, w_device=w_device)

    shape = images.get_shape().as_list()
    deconv4_op = deconvolution(deconv3_op, [batch_size,shape[1],shape[2],shape[3],1], "Deconv4", relu=False, ps_device=ps_device, w_device=w_device)

    return deconv4_op

def discriminator(images, size, keep_prob=1, batch_size=1, regularization_constant=0.0, ps_device="/cpu:0", w_device="/gpu:0"):

#   input: tensor of images
#   output: tensor of computed logits

    print_tensor_shape(images, "images discriminator")
# Convolution layer
    # with tf.variable_scope('Conv1'):

# weight variable 4d tensor, first two dims are patch (kernel) size       
# third dim is number of input channels and fourth dim is output channels

        # W_conv1 = tf.get_variable("W_conv1", shape=[3,3,3,1,64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
        # print_tensor_shape( W_conv1, 'W_conv1 shape')

        # conv1_op = tf.nn.conv3d( images, W_conv1, strides=[1,2,2,2,1], 
        #              padding="SAME", name='conv1_op' )
        # print_tensor_shape( conv1_op, 'conv1_op shape')

        # W_bias1 = tf.get_variable('W_bias1', shape=[1,17,17,17,64])

        # print_tensor_shape( W_bias1, 'W_bias1 shape')

        # bias1_op = conv1_op + W_bias1
        # print_tensor_shape( bias1_op, 'bias1_op shape')

        # relu1_op = tf.nn.relu( bias1_op, name='relu1_op' )
        # print_tensor_shape( relu1_op, 'relu1_op shape')

    relu1_op = convolution(images, 64, "Conv1", ps_device=ps_device, w_device=w_device)

# Conv layer
    # with tf.variable_scope('Conv2'):
    #     W_conv2 = tf.get_variable("W_conv2", shape=[3,3,3,64,128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
        
    #     print_tensor_shape( W_conv2, 'W_conv2 shape')

    #     conv2_op = tf.nn.conv3d( relu1_op, W_conv2, strides=[1,2,2,2,1],
    #                  padding="SAME", name='conv2_op' )

    #     print_tensor_shape( conv2_op, 'conv2_op shape')

    #     W_bias2 = tf.get_variable(name='W_bias2', shape=[1,9,9,9,128])
    #     print_tensor_shape( W_bias2, 'W_bias2 shape')

    #     bias2_op = conv2_op + W_bias2
    #     print_tensor_shape( bias2_op, 'bias2_op shape')

    #     relu2_op = tf.nn.relu( bias2_op, name='relu2_op' )
    #     print_tensor_shape( relu2_op, 'relu2_op shape')

    relu2_op = convolution(relu1_op, 128, "Conv2", ps_device=ps_device, w_device=w_device)
    
# Conv layer
    # with tf.variable_scope('Conv3'):
        
    #     W_conv3 = tf.get_variable("W_conv3", shape=[3,3,3,128,256], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
        
    #     print_tensor_shape( W_conv3, 'W_conv3 shape')

    #     conv3_op = tf.nn.conv3d( relu2_op, W_conv3, strides=[1,2,2,2,1],
    #                  padding='SAME', name='conv3_op' )
    #     print_tensor_shape( conv3_op, 'conv3_op shape')

    #     W_bias3 = tf.get_variable(name='W_bias3', shape=[1,5,5,5,256])

    #     print_tensor_shape( W_bias3, 'W_bias3 shape')

    #     bias3_op = conv3_op + W_bias3
    #     print_tensor_shape( bias3_op, 'bias3_op shape')

    #     relu3_op = tf.nn.relu( bias3_op, name='relu3_op' )
    #     print_tensor_shape( relu3_op, 'relu3_op shape')

    relu3_op = convolution(relu2_op, 256, "Conv3", ps_device=ps_device, w_device=w_device)
    
# Conv layer
    # with tf.name_scope('Conv4'):
    #     W_conv4 = tf.Variable(tf.truncated_normal([3,3,3,256,512],stddev=0.1,
    #                 dtype=tf.float32), name='W_conv4')
    #     print_tensor_shape( W_conv4, 'W_conv4 shape')

    #     conv4_op = tf.nn.conv3d( relu3_op, W_conv4, strides=[1,2,2,2,1],
    #                  padding='SAME', name='conv4_op' )
    #     print_tensor_shape( conv4_op, 'conv4_op shape')

    #     W_bias4 = tf.Variable( tf.zeros([1,5,5,5,512], dtype=tf.float32),
    #                       name='W_bias4')
    #     print_tensor_shape( W_bias4, 'W_bias4 shape')

    #     bias4_op = conv4_op + W_bias4
    #     print_tensor_shape( bias4_op, 'bias4_op shape')

    #     relu4_op = tf.nn.relu( bias4_op, name='relu4_op' )
    #     print_tensor_shape( relu4_op, 'relu4_op shape')

    relu4_op = convolution(relu3_op, 512, "Conv4", ps_device=ps_device, w_device=w_device)

    with tf.device(w_device):
        drop_op = tf.nn.dropout( relu4_op, keep_prob )
        print_tensor_shape( drop_op, 'drop_op shape' )

#Fully connected layer
    # with tf.variable_scope('Matmul5'):

    #     W_matmul5 = tf.get_variable("W_matmul5", shape=[5*5*5*256,512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
        
    #     print_tensor_shape( W_matmul5, 'W_matmul5 shape')        

    #     W_bias5 = tf.get_variable(name='W_bias5', shape=[512])                 
    
    with tf.device(w_device):
        shape = drop_op.get_shape().as_list()
        h_drop_op_flat = tf.reshape(drop_op, [-1, shape[1]*shape[2]*shape[3]*shape[4]])
    #     matmul5_op = tf.nn.relu(tf.matmul(h_drop_op_flat, W_matmul5) + W_bias5)

    matmul5_op = matmul(h_drop_op_flat, 512, "Matmul5", ps_device=ps_device, w_device=w_device)


    # with tf.variable_scope('Matmul6'):

    #     W_matmul6 = tf.get_variable("W_matmul6", shape=[512,2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
        
    #     print_tensor_shape( W_matmul6, 'W_matmul6 shape')        

    #     W_bias6 = tf.get_variable(name='W_bias6', shape=[2])

    #     matmul6_op = tf.matmul(matmul5_op, W_matmul6) + W_bias6

    matmul6_op = matmul(matmul5_op, 2, "Matmul6", ps_device=ps_device, w_device=w_device)

    return matmul6_op


def loss(logits, labels):

    print_tensor_shape( logits, 'logits shape')
    print_tensor_shape( labels, 'labels shape')

    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy')

    loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

    return loss

def training_adam(loss, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False, name='Adam', var_list=None):

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        use_locking=use_locking,
        name=name)

    train_op = optimizer.minimize(loss, var_list=var_list)

    return train_op


def training(loss, learning_rate, decay_steps, decay_rate, name):
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
    lr = tf.train.exponential_decay(learning_rate,
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

def evaluation(logits, labels, name="accuracy"):
    values, indices = tf.nn.top_k(labels, 1);
    correct = tf.reshape(tf.nn.in_top_k(logits, tf.cast(tf.reshape( indices, [-1 ] ), tf.int32), 1), [-1] )
    print_tensor_shape( correct, 'correct shape')
    return tf.reduce_mean(tf.cast(correct, tf.float32), name=name)

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
