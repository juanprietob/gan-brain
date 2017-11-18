
"""Converts NRRD data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import nrrd
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def saveTFRecord(filename, writer, class_writer, label):

  print("Reading:", filename)

  img, head = nrrd.read(filename)
  img = img.astype(np.float32)
  
  img_sizes = head["sizes"]

  height = img_sizes[0]
  width = img_sizes[1]
  depth = img_sizes[2]

  img_raw = img.tostring()  

  example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(height),
      'width': _int64_feature(width),
      'depth': _int64_feature(depth),
      'raw': _bytes_feature(img_raw),
      'label': _int64_feature(label)
    }))

  writer.write(example.SerializeToString())
  class_writer.write(example.SerializeToString())



def main(unused_argv):
  # Get the data.

  writer = tf.python_io.TFRecordWriter(FLAGS.output)

  dirs = [ name for name in os.listdir(FLAGS.directory) if os.path.isdir(os.path.join(FLAGS.directory, name)) ]
  
  label = -1
  for d in dirs:
    label+=1
    images = os.listdir(os.path.join(FLAGS.directory, d))

    class_writer = tf.python_io.TFRecordWriter(os.path.join(os.path.dirname(FLAGS.output), d + ".tfRecords"))

    for img in images:
      saveTFRecord(os.path.join(FLAGS.directory, d, img), writer, class_writer, label)

    class_writer.close();


  writer.close()

  print("TFRecords:", FLAGS.output)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      required=True,
      help='The directory contains a directory for each class. The class directories contain nrrd image files'
  )
  parser.add_argument(
      '--output',
      type=str,
      required=True,
      help='Output filename for output tfRecords.'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=5000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )  
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)