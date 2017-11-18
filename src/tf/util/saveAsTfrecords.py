import nrrd
import tensorflow as tf
import os
import numpy as np


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('directory', '../data/gan/V24/', 'Data directory.')
flags.DEFINE_string('saving_directory', '../data/gan/tfrecords/', 'Data directory.')
flags.DEFINE_string('filename', 'train.tfrecords', 'filename')

filename = FLAGS.filename
directory = FLAGS.directory
saving_directory = FLAGS.saving_directory


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_im(path):
    images = sorted(os.listdir(path))
    nb_images = int(len(images))
    data = np.ndarray(shape=(int(nb_images/2), 195, 233, 2), dtype=np.float32)

    num = 0
    ni = 0

    for ima in images:
        ima = os.path.join(path, ima)
        # print(ima)
        # print('ni : ', ni)
        # print('num image : ', num)

        im_nrrd, _ = nrrd.read(ima)
        im_nrrd = im_nrrd.astype(np.float32)
        im_nrrd = im_nrrd.reshape((195,233))

        data[num, :, :, ni] = im_nrrd

        if ni == 0: ni = 1
        else:
            ni = 0
            num += 1
    return data, int(nb_images/2)


def get_la(path):
    labs = sorted(os.listdir(path))
    nb_labs = int(len(labs))
    data = np.ndarray(shape=(int(nb_labs), 195, 233, 1), dtype=np.float32)

    num = 0

    for ima in labs:
        ima = os.path.join(path, ima)
        # print(ima)
        # print('num label : ', num)

        im_nrrd, _ = nrrd.read(ima)
        im_nrrd = im_nrrd.astype(np.float32)
        im_nrrd = im_nrrd.reshape((195, 233))

        data[num, :, :, 0] = im_nrrd
        num += 1

    return data

im, nb_im = get_im(os.path.join(directory, 'V06'))
la = get_la(os.path.join(directory, 'V24_t1'))

writer = tf.python_io.TFRecordWriter(os.path.join(saving_directory, filename))


for i in xrange(nb_im):
    img = im[i]
    labl = la[i]

    # The reason to store image sizes was demonstrated
    # in the previous example -- we have to know sizes
    # of images to later read raw serialized string,
    # convert to 1d array and convert to respective
    # shape that image used to have.
    height = img.shape[0]
    width = img.shape[1]
    depth = img.shape[2]

    img_raw = img.tostring()
    labl_raw = labl.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'depth': _int64_feature(depth),
        'V24_raw': _bytes_feature(labl_raw),
        'V06_raw': _bytes_feature(img_raw)}))

    writer.write(example.SerializeToString())

writer.close()


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'V24_raw': tf.FixedLenFeature([], tf.string),
            'V06_raw': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['V06_raw'], tf.float32)
    label = tf.decode_raw(features['V24_raw'], tf.float32)

    # Set image and label shapes
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)

    image_shape = tf.pack([195, 233, 2])
    label_shape = tf.pack([195, 233,1])

    image = tf.reshape(image, image_shape)
    label = tf.reshape(label, label_shape)

    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=3,
                                            capacity=300,
                                            num_threads=2,
                                            min_after_dequeue=5)

    return images, labels


filename_queue = tf.train.string_input_producer(
    [os.path.join(directory,filename)], num_epochs=1)

# Even when reading in multiple threads, share the filename
# queue.
image, label = read_and_decode(filename_queue)

# The op for initializing the variables.
init_op = tf.group(tf.initialize_all_variables(),
                   tf.initialize_local_variables())

with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Let's read off 3 batches just for example
    for i in xrange(3):
        img, lab = sess.run([image, label])
        print(img[0, :, :, :].shape)
        print(lab.shape)
        print(np.amin(lab))
        print('current batch')

        # We selected the batch size of two
        # So we should get two image pairs in each batch
        # Let's make sure it is random

        # nrrd.write('prediction_a_' + i.__str__() + '.nrrd', img[0, :, :, 0])
        # nrrd.write('l_a_' + i.__str__() + '.nrrd', lab[0, :, :, 0])

    coord.request_stop()
    coord.join(threads)
