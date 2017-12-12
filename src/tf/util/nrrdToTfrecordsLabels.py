
"""Converts NRRD data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import glob

import tensorflow as tf
import nrrd
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def saveTFRecord(filename, filenamelabel, writer):

  print("Reading:", filename)

  img, head = nrrd.read(filename)
  img = img.astype(np.float32)

  print("Reading:", filenamelabel)

  imglabel, headlabel = nrrd.read(filenamelabel)
  imglabel = imglabel.astype(np.float32)
  
  img_sizes = head["sizes"]
  label_sizes = headlabel["sizes"]

  if img_sizes[0] != label_sizes[0] or img_sizes[1] != label_sizes[1] or img_sizes[2] != label_sizes[2]:
    print("Sizes in files:", filename, filenamelabel, "have different dimensions. Skipping...")
  else:
    height = img_sizes[0]
    width = img_sizes[1]
    depth = img_sizes[2]

    img_raw = img.tostring()
    imglabel_raw = imglabel.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'depth': _int64_feature(depth),
        'raw': _bytes_feature(img_raw),
        'label': _bytes_feature(imglabel_raw)
      }))

    writer.write(example.SerializeToString())



def main(unused_argv):
  # Get the data.

  print("Writing", FLAGS.output)
  writer = tf.python_io.TFRecordWriter(FLAGS.output)

  #label+=1
  images = [ name for name in glob.glob(os.path.join(FLAGS.directory, "*.nrrd")) if "_label.nrrd" not in name ]
  test_size = int(len(images)*(1. - FLAGS.validation_size))

  p = np.random.permutation(len(images))
  images = np.array(images)[p]

  train_images = images[0:test_size]
  test_images = images[test_size:]

  for img in train_images:
    
    labelimg = os.path.splitext(img)[0] + "_label.nrrd"
    saveTFRecord(os.path.join(FLAGS.directory, img), os.path.join(FLAGS.directory, labelimg), writer)
  
  writer.close()

  outputtest = os.path.splitext(FLAGS.output)[0] + "_test.tfRecords"

  print("Writing", outputtest)
  writer = tf.python_io.TFRecordWriter(outputtest)
  
  for img in test_images:
    
    labelimg = os.path.splitext(img)[0] + "_label.nrrd"
    saveTFRecord(os.path.join(FLAGS.directory, img), os.path.join(FLAGS.directory, labelimg), writer)

  writer.close()


  print("Total images:", len(images))
  print("Train images:", len(train_images))
  print("Test images:", len(test_images))
  print("TFRecords:", FLAGS.output)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      required=True,
      help='The directory contains nrrd image files. There are pairs <some image filename>.nrrd and label images have suffix <some image filename>_label.nrrd, check the sampleImage executable to create the samples.'
  )
  parser.add_argument(
      '--output',
      type=str,
      required=True,
      help='Output filename for output tfRecords.'
  )
  parser.add_argument(
      '--validation_size',
      type=float,
      default=0.2,
      help="Divide the data for validation using this ratio"
  )  
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)