from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


# function to print the tensor shape.  useful for debugging
BLUE = "\033[0;34m"
BLUE_BOLD = "\033[1;34m"
RED = "\033[0;31m"
RED_BOLD = "\033[1;31m"
GREEN = "\033[0;32m"
GREEN_BOLD = "\033[1;32m"
YELLOW = "\033[0;33m"
YELLOW_BOLD = "\033[1;33m"
CYAN = "\033[0;36m"
CYAN_BOLD =  "\033[1;36m"
NC_BOLD = "\033[1m"
NC = "\033[0m"


def print_tensor_shape(tensor, string):
    # input: tensor and string to describe it
    if __debug__:
        print(BLUE_BOLD + 'DEBUG ' + NC + string, tensor.get_shape())

def shuffle_batch(a, b, size):    
    p = np.random.permutation(size)
    return a[p], b[p]


def getSize(shape, nb_div):
    res = shape

    for i in xrange(int(nb_div)):
        res[0] = int(np.ceil(res[0] / 2))
        res[1] = int(np.ceil(res[1] / 2))
        # res = tf.ceil(tf.div(res, 2))

    return res


def fullc(inp, shape, nb_feats, nb_fullc, iftrain, info_file, drop=False, relu=True, name=''):
    with tf.name_scope('Fullyconnected' + nb_fullc):
        W_fc = tf.Variable(tf.truncated_normal([shape, nb_feats], stddev=0.1,
                                               dtype=tf.float32), name=name + 'W_fc' + nb_fullc)
        B_fc = tf.Variable(tf.constant(0.1, shape=[nb_feats], name=name + 'B_fc' + nb_fullc))

        resh_op = tf.reshape(inp, [-1, shape])
        fc_op = tf.nn.relu(tf.matmul(resh_op, W_fc) + B_fc)

        print_tensor_shape(W_fc, GREEN + 'W_fc' + nb_fullc + ' shape' + NC)
        print_tensor_shape(fc_op, ' fc' + nb_fullc + '_op shape')

        if iftrain:
            f = open(info_file, 'ab')
            f.write('W_fc' + nb_fullc + ' shape' + W_fc.get_shape().__str__() + '\n')
            f.write('fc' + nb_fullc + '_op shape' + fc_op.get_shape().__str__() + '\n')

        if relu:
            relu_op = tf.nn.relu(fc_op, name='relu' + nb_fullc + '_op')
            print_tensor_shape(relu_op, ' relu' + nb_fullc + '_op shape')
            if iftrain:
                f.write('relu' + nb_fullc + '_op shape ' + relu_op.get_shape().__str__() + '\n')

            if drop:
                drop_op = tf.nn.dropout(relu_op, 1.0)
                print_tensor_shape(drop_op, ' drop_op shape')
                if iftrain:
                    f.write('drop_op shape ' + drop_op.get_shape().__str__() + '\n')
                relu_op = drop_op

            fc_op = relu_op

        if iftrain:
            f.write('\n')
            f.close()

    return fc_op


def conv_2(inp, filt_settings, strides, nb_conv, iftrain, info_file, drop=False, relu=True, name=''):
    with tf.name_scope('Conv' + nb_conv):
        W_conv = tf.Variable(tf.truncated_normal(filt_settings, stddev=0.1,
                                                 dtype=tf.float32), name=name + 'W_conv' + nb_conv)

        B_conv = tf.Variable(tf.constant(0.1, shape=[filt_settings[-1]], name=name + 'B_conv' + nb_conv))

        conv_op = tf.nn.conv2d(inp, W_conv, strides=strides,
                               padding="SAME", name='conv' + nb_conv + '_op') + B_conv

        print_tensor_shape(W_conv, GREEN + 'W_conv' + nb_conv + ' shape' + NC)
        print_tensor_shape(conv_op,' conv' + nb_conv + '_op shape')

        if iftrain:
            f = open(info_file, 'ab')
            f.write('W_conv' + nb_conv + ' shape' + W_conv.get_shape().__str__() + '\n')
            f.write('conv' + nb_conv + '_op shape' + conv_op.get_shape().__str__() + '\n')

        if relu:
            relu_op = tf.nn.relu(conv_op, name='relu' + nb_conv + '_op')
            print_tensor_shape(relu_op, ' relu' + nb_conv + '_op shape')
            if iftrain:
                f.write('relu' + nb_conv + '_op shape ' + relu_op.get_shape().__str__() + '\n')

            if drop:
                drop_op = tf.nn.dropout(relu_op, 1.0)
                print_tensor_shape(drop_op, ' drop_op shape')
                if iftrain:
                    f.write('drop_op shape ' + drop_op.get_shape().__str__() + '\n')
                relu_op = drop_op

            conv_op = relu_op

        if iftrain:
            f.write('\n')
            f.close()

    return conv_op


def pool_2(inp, ks, strides, nb_pool, iftrain, info_file):
    with tf.name_scope('Pool' + nb_pool):
        pool_op = tf.nn.max_pool(inp, ksize=ks,
                                 strides=strides, padding='SAME')
        print_tensor_shape(pool_op, YELLOW + ' pool' + nb_pool + '_op shape' + NC)

        if iftrain:
            f = open(info_file, 'ab')
            f.write('pool' + nb_pool + '_op shape' + pool_op.get_shape().__str__() + '\n')
            f.write('\n')
            f.close()

    return pool_op


def avg_pool_2(inp, ks, strides, nb_avg_pool, iftrain, info_file):
    with tf.name_scope('Avg_Pool' + nb_avg_pool):
        pool_op = tf.nn.avg_pool(inp, ksize=ks,
                                 strides=strides, padding='SAME')
        print_tensor_shape(pool_op, YELLOW + ' avg_pool' + nb_avg_pool + '_op shape' + NC)

        if iftrain:
            f = open(info_file, 'ab')
            f.write('avg_pool' + nb_avg_pool + '_op shape' + pool_op.get_shape().__str__() + '\n')
            f.write('\n')
            f.close()

    return pool_op


def batch_norm(inp, n_out, nb_norm, iftrain, info_file, name=''):
    with tf.name_scope('Norm' + nb_norm):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name=name + 'beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name=name + 'gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inp, [0, 1, 2], name=name + 'moments')
        normed = tf.nn.batch_normalization(inp, batch_mean, batch_var, beta, gamma, 1e-3, name=name + 'bn' + nb_norm)

        print_tensor_shape(normed, ' norm' + nb_norm + '_op shape')

        if iftrain:
            f = open(info_file, 'ab')
            f.write('norm' + nb_norm + '_op shape' + normed.get_shape().__str__() + '\n')
            f.write('\n')
            f.close()

    return normed


def conv_up2(inp, filt_settings, output_shape, strides, nb_convup, iftrain, info_file, name=''):
    with tf.name_scope('Upscore' + nb_convup):
        W_upscore = tf.Variable(tf.truncated_normal(filt_settings,
                                                    stddev=0.1, dtype=tf.float32), name=name + 'W_upscore_' + nb_convup)
        upscore_conv_op = tf.nn.conv2d_transpose(inp,
                                                 W_upscore,
                                                 output_shape=output_shape,
                                                 strides=strides,
                                                 padding='SAME', name='upscore_conv' + nb_convup + '_op')

        print_tensor_shape(W_upscore, GREEN + 'W_upscore_' + nb_convup + ' shape' + NC)
        print_tensor_shape(upscore_conv_op, ' upscore_conv' + nb_convup + '_op shape')

        if iftrain:
            f = open(info_file, 'ab')
            f.write('W_upscore' + nb_convup + ' shape' + W_upscore.get_shape().__str__() + '\n')
            f.write('upscore_conv_op' + nb_convup + '_op shape' + upscore_conv_op.get_shape().__str__() + '\n')
            f.write('\n')
            f.close()

    return upscore_conv_op


def inception(inp, f, f_a, f_b_1, f_b_2, f_c_1, f_c_2, f_d, nb_conv, nb_pool, nb_incep, iftrain, info_file):
    with tf.name_scope('Inception' + nb_incep):
        op_a = conv_2(inp, [1, 1, f, f_a], [1, 1, 1, 1], nb_conv.__str__(), iftrain, info_file)

        op_b_1 = conv_2(inp, [1, 1, f, f_b_1], [1, 1, 1, 1], (nb_conv + 1).__str__(), iftrain, info_file)
        op_b_2 = conv_2(op_b_1, [3, 3, f_b_1, f_b_2], [1, 1, 1, 1], (nb_conv + 2).__str__(), iftrain, info_file)

        op_c_1 = conv_2(inp, [1, 1, f, f_c_1], [1, 1, 1, 1], (nb_conv + 3).__str__(), iftrain, info_file)
        op_c_2 = conv_2(op_c_1, [5, 5, f_c_1, f_c_2], [1, 1, 1, 1], (nb_conv + 4).__str__(), iftrain, info_file)

        op_d_1 = pool_2(inp, [1, 3, 3, 1], [1, 1, 1, 1], nb_pool, iftrain, info_file)
        op_d_2 = conv_2(op_d_1, [1, 1, f, f_d], [1, 1, 1, 1], (nb_conv + 5).__str__(), iftrain, info_file)

        concat = tf.concat(3, [op_a, op_b_2, op_c_2, op_d_2])
        print_tensor_shape(concat, RED_BOLD + ' incept' + nb_incep + '_op shape' + NC)

    return concat













