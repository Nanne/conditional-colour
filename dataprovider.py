from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys, json

from six.moves import urllib
import tensorflow as tf
import matplotlib

FLAGS = tf.app.flags.FLAGS

# Size of the crop:
IMAGE_SIZE = 224

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
TEST_FILE = 'test.tfrecords'

def getsize(filename):
    jsfile = filename + ".json"
    if tf.gfile.Exists(jsfile):
        with open(jsfile, 'r') as f:
            N = json.load(f)['count']
    else:
        N = 0
        for record in tf.python_io.tf_record_iterator(filename):
            N += 1
        with open(jsfile, 'w') as f:
            f.write(json.dumps({'count': N}))
    return N

def read_record(filename_queue):
    """Reads TF record.

    Args:
        filename_queue: A queue of strings with the filenames to read from.

    Returns:
        An object representing a single example, with the following fields:
            height: number of rows in the result
            width: number of columns in the result
            depth: number of color channels in the result (3)
            label: an int64 Tensor with the image label.
            id: an int64 Tensor with the image id.
            image: a [height, width, depth] float64 Tensor with the image data
    """

    class DataRecord(object):
        pass
    result = DataRecord()

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'id': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
            })

    image = tf.decode_raw(features['image_raw'], tf.float64)
    result.height = tf.cast(features['height'], tf.int32)
    result.width = tf.cast(features['width'], tf.int32)
    result.depth = tf.cast(features['depth'], tf.int32)
 
    im_shape = tf.stack([result.height, result.width, result.depth])
    result.image = tf.reshape(image, im_shape)
   
    result.label = features['label']
    result.id = features['id']


    return result


def _generate_batch(L, AB, label, min_queue_examples,
                                    batch_size, shuffle, size=0):
    """Construct a queued batch.

    Args:
        L: 3-D Tensor of [height, width, 1] of type.float32.
        AB: 3-D Tensor of [height, width, 2] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
            in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: labels. 2D tensor of [batch_size, 1] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        Ls, ABs, labels = tf.train.shuffle_batch(
            [L, AB, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        Ls, ABs, labels = tf.train.batch(
            [L, AB, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    if size > 0:
        return Ls, ABs, labels, size
    else:
        return Ls, ABs, labels

def distorted_inputs(get_size=False):
    """Construct distorted input training using the Reader ops.

    Args:

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: labels. 2D tensor of [batch_size, 1] size.
    """
    data_dir = FLAGS.data_dir
    batch_size = FLAGS.batch_size

    filename = os.path.join(data_dir, TRAIN_FILE)

    if get_size:
        N = getsize(filename)
    
    filename_queue = tf.train.string_input_producer([filename])

    read_input = read_record(filename_queue)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Randomly crop a [height, width] section of the image.
    image = tf.random_crop(read_input.image, [height, width, 3])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32)

    L,A,B = tf.split(axis=2, num_or_size_splits=3, value=image) 
    # FIXME: clipping isn't very nice but anything clipped should still fall in the same bin as non-clipped
    # without clipping the quantization in the loss should probably be more proper
    AB = tf.clip_by_value(tf.concat(values=[A,B], axis=2), -1.0, 1.0)

    # delta of 0.016 should give a CIEDE2000 of less than 1
    L = tf.image.random_brightness(L, 0.016)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.3
    if get_size:
        min_queue_examples = max(100, int(min(10000, N) *
                           min_fraction_of_examples_in_queue))
    else:
        min_queue_examples = int(10000 *
                           min_fraction_of_examples_in_queue)
    print ('Filling queue with %d images before starting to train. '
         'This might take a few minutes.' % min_queue_examples)

    # Generate a batch by building up a queue of examples.
    return _generate_batch(L, AB, read_input.label,
                                     min_queue_examples, batch_size,
                                     shuffle=True, size=N)

def inputs(get_size=False):
    """Construct input for evaluation using the Reader ops.

    Args:

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: labels. 2D tensor of [batch_size, 1] size.
    """

    data_dir = FLAGS.data_dir
    batch_size = FLAGS.batch_size

    filename = os.path.join(data_dir, VALIDATION_FILE)

    N = 0
    if get_size:
        N = getsize(filename)

    filename_queue = tf.train.string_input_producer([filename])

    read_input = read_record(filename_queue)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    image = tf.image.resize_image_with_crop_or_pad(read_input.image,
                                                         width, height)
    image = tf.reshape(image, (width,height,3))
    #image = tf.random_crop(read_input.image, [height, width, 3])
    image = tf.cast(image, tf.float32)

    L,A,B = tf.split(axis=2, num_or_size_splits=3, value=image) 
    AB = tf.clip_by_value(tf.concat(values=[A,B], axis=2), -1.0, 1.0)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    if get_size:
        min_queue_examples = max(100, int(min(10000, N) *
                           min_fraction_of_examples_in_queue))
    else:
        min_queue_examples = int(10000 *
                           min_fraction_of_examples_in_queue)

    # Generate a batch by building up a queue of examples.
    return _generate_batch(L, AB, read_input.label,
                                 min_queue_examples, batch_size,
                                     shuffle=False, size=N)

def input(source = 'val', get_size=False):
    """Construct input for evaluation using the Reader ops.

    Args:

    Returns:
        image: Image. 4D tensor of [1, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        label: label. 1D tensor of [1] size.
    """

    data_dir = FLAGS.data_dir
    batch_size = FLAGS.batch_size

    sourcefile = VALIDATION_FILE
    if source.lower() == 'train':
        sourcefile = TRAIN_FILE
    elif source.lower() == 'test':
        sourcefile = TEST_FILE

    filename = os.path.join(data_dir, sourcefile)
    N = 0
    if get_size:
        N = getsize(filename)

    filename_queue = tf.train.string_input_producer([filename])

    read_input = read_record(filename_queue)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    #image = tf.random_crop(read_input.image, [height, width, 3])
    image = tf.image.resize_image_with_crop_or_pad(read_input.image,
                                                         width, height)
    image = tf.reshape(image, (1, height, width, 3))
    image = tf.cast(image, tf.float32)

    L,A,B = tf.split(axis=3, num_or_size_splits=3, value=image) 
    AB = tf.clip_by_value(tf.concat(values=[A,B],axis=3), -1.0, 1.0)

    label = tf.reshape(read_input.label, (1,))

    if get_size:
        return L, AB, label, N
    else:
        return L, AB, label
