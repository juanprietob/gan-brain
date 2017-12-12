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
            'label': tf.FixedLenFeature([], tf.string)
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
    label = tf.decode_raw(features['label'], tf.float32)    
    label_re = tf.reshape(label, (size))
    print_tensor_shape(label_re, namescope + ' image label')

    return image_re, label_re


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

        print_tensor_shape(images, namescope)
        # labels = tf.one_hot(labels, 2)
        print_tensor_shape(labels, namescope + ' labels')


        # beta = tf.Variable(tf.constant(0.0, shape=[1]),
        #                              name='beta', trainable=True)
        # gamma = tf.Variable(tf.constant(1.0, shape=[1]),
        #                               name='gamma', trainable=True)
        # mean, variance = tf.nn.moments(images, [0])
        # images = tf.nn.batch_normalization(images, mean, variance, beta, gamma, 1e-3)

        return images, labels

def convolution2d(images, out_channels, name, relu=True, ps_device="/cpu:0", w_device="/gpu:0"):

    in_channels = images.get_shape().as_list()[-1]

    with tf.variable_scope(name):
# weight variable 4d tensor, first two dims are patch (kernel) size       
# third dim is number of input channels and fourth dim is output channels
        with tf.device(ps_device):

            w_conv_name = 'w_' + name
            w_conv = tf.get_variable(w_conv_name, shape=[3,3,in_channels,out_channels], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
            print_tensor_shape( w_conv, 'weight shape')

            b_conv_name = 'b_' + name
            b_conv = tf.get_variable(b_conv_name, shape=[out_channels])
            print_tensor_shape( b_conv, 'bias shape')

        with tf.device(w_device):
            conv_op = tf.nn.conv2d( images, w_conv, strides=[1,2,2,1], padding="SAME", name='conv1_op')
            print_tensor_shape( conv_op, 'conv_op shape')

            conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

            if(relu):
                conv_op = tf.nn.relu( conv_op, name='relu_op' ) 
                print_tensor_shape( conv_op, 'relu_op shape')

            return conv_op

def convolution(images, name, activation=None, out_channels=1, ps_device="/cpu:0", w_device="/gpu:0", w_shape=None, strides=None, padding='SAME'):

    in_channels = images.get_shape().as_list()[-1]

    if w_shape is None:
        w_shape = [5,5,5,in_channels,out_channels]

    if strides is None:
        strides = [1,2,2,2,1]

    with tf.variable_scope(name):
# weight variable 4d tensor, first two dims are patch (kernel) size       
# third dim is number of input channels and fourth dim is output channels
        with tf.device(ps_device):

            w_conv_name = 'w_' + name
            w_conv = tf.get_variable(w_conv_name, shape=w_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
            print_tensor_shape( w_conv, name + ' weight shape')

            b_conv_name = 'b_' + name
            b_conv = tf.get_variable(b_conv_name, shape=w_shape[-1:])
            print_tensor_shape( b_conv, name + ' bias shape')

        with tf.device(w_device):
            conv_op = tf.nn.conv3d( images, w_conv, strides=strides, padding=padding, name='conv1_op')
            print_tensor_shape( conv_op, name + ' conv_op shape')

            conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

            if(activation):
                conv_op = activation( conv_op, name='activation_op' ) 
                print_tensor_shape( conv_op, 'activation_op shape')

            return conv_op

def deconvolution2d(images, output_shape, name, activation=None, ps_device="/cpu:0", w_device="/gpu:0", w_shape=None, strides=None, padding="SAME"):

    with tf.variable_scope(name):

        in_channels = images.get_shape()[-1]

        out_channels = output_shape[-1]

        if w_shape is None:
            w_shape = [3,3,in_channels,out_channels]

        if strides is None:
            strides = [1,2,2,1]

        with tf.device(ps_device):
            w_deconv_name = 'w_' + name
            w_deconv = tf.get_variable(w_deconv_name, shape=w_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
            print_tensor_shape( w_deconv, name + 'weight shape')

            b_deconv_name = 'b_' + name
            b_deconv = tf.get_variable(b_deconv_name, shape=[out_channels])
            print_tensor_shape( b_deconv, name + 'bias shape')

        with tf.device(w_device):

            deconv_op = tf.nn.conv2d_transpose( images, w_deconv, 
                output_shape=output_shape,
                # use_bias=True,
                strides=strides,
                padding=padding, name='deconv_op' )

            print_tensor_shape( deconv_op, 'deconv_op shape')

            deconv_op = tf.nn.bias_add(deconv_op, b_deconv, name='bias_add_op')

        if activation:
            deconv_op = activation( deconv_op, name='activation_op' )
            print_tensor_shape( deconv_op, 'activation_op shape')

        return deconv_op

def deconvolution(images, output_shape, name, activation=None, ps_device="/cpu:0", w_device="/gpu:0", w_shape=None, strides=None, padding="SAME"):

    with tf.variable_scope(name):

        in_channels = images.get_shape()[-1]

        out_channels = output_shape[4]

        if w_shape is None:
            w_shape = [5,5,5,out_channels,in_channels]

        if strides is None:
            strides = [1,2,2,2,1]

        with tf.device(ps_device):
            w_deconv_name = 'w_' + name
            w_deconv = tf.get_variable(w_deconv_name, shape=w_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
            print_tensor_shape( w_deconv, name + ' weight shape')

            b_deconv_name = 'b_' + name
            b_deconv = tf.get_variable(b_deconv_name, shape=[out_channels])
            print_tensor_shape( b_deconv, name + ' bias shape')

        with tf.device(w_device):

            deconv_op = tf.nn.conv3d_transpose( images, w_deconv, 
                output_shape=output_shape,
                # use_bias=True,
                strides=strides,
                padding=padding, name='deconv_op' )

            print_tensor_shape( deconv_op, 'deconv_op shape')

            deconv_op = tf.nn.bias_add(deconv_op, b_deconv, name='bias_add_op')

        if activation:
            deconv_op = activation( deconv_op, name='activation_op' )
            print_tensor_shape( deconv_op, 'activation_op shape')

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


def generator(images, keep_prob=1, batch_size=1, regularization_constant=0.0, ps_device="/cpu:0", w_device="/gpu:0", is_training=False):

# Encoder part of the network

#   input: tensor of images
#   output: tensor of computed logits

# resize the image tensors to add the number of channels, 1 in this case
# required to pass the images to various layers upcoming in the graph
    #print("Image size:", size)
    #num_channels = size[0], depth = size[0], height = size[1], width = size[2], num_channels = size[3]
    
    
    images = tf.layers.batch_normalization(images, training=is_training)
    
    print_tensor_shape(images, "images")
# Convolution layer

    conv1_op = convolution(images, "Conv1", out_channels=256, activation=tf.nn.relu, ps_device=ps_device, w_device=w_device, padding="SAME")

    conv2_op = convolution(conv1_op, "Conv2", out_channels=512, activation=tf.nn.relu,ps_device=ps_device, w_device=w_device, padding="SAME")

    # conv3_op = convolution(conv2_op, "Conv3", out_channels=1024, activation=tf.nn.relu, ps_device=ps_device, w_device=w_device, padding="VALID")

    # conv4_op = convolution(conv3_op, "Conv4", out_channels=1280, activation=tf.nn.relu, ps_device=ps_device, w_device=w_device, padding="VALID")

    #relu4_op = convolution(relu3_op, 4048, "Conv4", ps_device=ps_device, w_device=w_device)

    # shape = conv3_op.get_shape().as_list()
    # deconv1_op = deconvolution(conv4_op, shape, "Deconv1", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device, padding="VALID")

    # shape = conv2_op.get_shape().as_list()
    # deconv2_op = deconvolution(conv3_op, shape, "Deconv2", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device, padding="VALID")
    
    shape = conv1_op.get_shape().as_list()
    deconv3_op = deconvolution(conv2_op, shape, "Deconv3", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device, padding="SAME")

    shape = images.get_shape().as_list()
    shape[4] = 128
    deconv4_op = deconvolution(deconv3_op, shape, "Deconv4", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device, padding="SAME")

    with tf.device(w_device):
        deconv4_op = tf.nn.dropout( deconv4_op, keep_prob )

    deconv4_op = tf.concat([images, deconv4_op], axis=4)
    convp_op = convolution(deconv4_op, "ConvScore", strides=[1, 1, 1, 1, 1], w_shape=[1, 1, 1, 129, 1], ps_device=ps_device, w_device=w_device, padding="SAME")
    

    return convp_op

def generator2d(images, keep_prob=1, batch_size=1, regularization_constant=0.0, ps_device="/cpu:0", w_device="/gpu:0", is_training=False):

# Encoder part of the network

#   input: tensor of images
#   output: tensor of computed logits

# resize the image tensors to add the number of channels, 1 in this case
# required to pass the images to various layers upcoming in the graph
    #print("Image size:", size)
    #num_channels = size[0], depth = size[0], height = size[1], width = size[2], num_channels = size[3]
    
    images = tf.layers.batch_normalization(images, training=is_training)
    
    print_tensor_shape(images, "images")
# Convolution layer

    relu1_op = convolution2d(images, 128, "Conv1", ps_device=ps_device, w_device=w_device)

    relu2_op = convolution2d(relu1_op, 512, "Conv2", ps_device=ps_device, w_device=w_device)

    with tf.device(w_device):
        relu2_op = tf.nn.dropout( relu2_op, keep_prob )

    relu3_op = convolution2d(relu2_op, 2048, "Conv3", ps_device=ps_device, w_device=w_device)

    #relu4_op = convolution(relu3_op, 4048, "Conv4", ps_device=ps_device, w_device=w_device)

    shape = relu2_op.get_shape().as_list()
    deconv1_op = deconvolution2d(relu3_op, shape, "Deconv1", ps_device=ps_device, w_device=w_device)

    shape = relu1_op.get_shape().as_list()
    deconv2_op = deconvolution2d(deconv1_op, shape, "Deconv2", ps_device=ps_device, w_device=w_device)

    with tf.device(w_device):
        deconv2_op = tf.nn.dropout( deconv2_op, keep_prob )
    
    shape = images.get_shape().as_list()
    deconv3_op = deconvolution2d(deconv2_op, shape, "Deconv3", relu=False, ps_device=ps_device, w_device=w_device)

    #shape = images.get_shape().as_list()
    #deconv4_op = deconvolution(deconv3_op, [batch_size,shape[1],shape[2],shape[3],1], "Deconv4", relu=False, ps_device=ps_device, w_device=w_device)

    return deconv3_op

def loss(logits, labels):

    print_tensor_shape( logits, 'logits shape')
    print_tensor_shape( labels, 'labels shape')

    #labels = tf.to_int64(labels)

    #loss = tf.losses.absolute_difference(predictions=logits, labels=labels)
    loss = tf.losses.mean_squared_error(predictions=logits, labels=labels)
    #loss = tf.losses.mean_pairwise_squared_error(predictions=logits, labels=labels)
    #loss = tf.losses.huber_loss(predictions=logits, labels=labels, delta=10.0)

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
    
    # accuracy = tf.metrics.accuracy(labels=labels, predictions=logits, name=name)
    # tf.summary.scalar("accuracy", accuracy[0])
    # return accuracy
    accuracy = tf.metrics.root_mean_squared_error(labels=labels, predictions=logits, name=name)
    tf.summary.scalar("accuracy", accuracy[0])

    return accuracy


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
