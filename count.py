"""Count occurances of colours as quantized bins for A and B channels of LAB images"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time, math
import os.path
from os.path import join
import numpy as np
import tensorflow as tf

import dataprovider
import config

import h5py

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('out_dir', 'counts/',
                           """Directory where to write the counts to.""")
tf.app.flags.DEFINE_integer('num_examples', 92580,
                            """Number of examples to run.""")

def quantize(ab):
    a = tf.reshape(ab[:,:,:,0], (-1,))
    b = tf.reshape(ab[:,:,:,1], (-1,))

    # quantize:
    q_a = tf.one_hot(tf.to_int32((a+1) / (2/FLAGS.quantiles)), FLAGS.quantiles)
    q_b = tf.one_hot(tf.to_int32((b+1) / (2/FLAGS.quantiles)), FLAGS.quantiles)

    return tf.reduce_sum(q_a, 0), tf.reduce_sum(q_b, 0)

def count(device = '/gpu:2'):
    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)

    f = h5py.File(join(FLAGS.out_dir, "counts.hdf5"), "w")
    A_count_dset = f.create_dataset("Acounts", (FLAGS.num_examples,32), dtype='float32')
    B_count_dset = f.create_dataset("Bcounts", (FLAGS.num_examples,32), dtype='float32')
    label_dset = f.create_dataset("labels", (FLAGS.num_examples,1), dtype='int32')

    with tf.Graph().as_default() as g:
        with tf.device(device):
            _, AB, label = dataprovider.input('train')
            qA, qB = quantize(AB) 
            
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                                                     start=True))

                step = 0
                true_count = 0
                while step < FLAGS.num_examples and not coord.should_stop():
                    A_count_dset[step], B_count_dset[step], label_dset[step] = sess.run([qA, qB, label])

                    step += 1
                    if step % 100 == 0:
                        print('processing %d/%d (%.2f%% done)' % (step, FLAGS.num_examples, step*100.0/FLAGS.num_examples))

            except Exception as e:    # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

    f.close()

def main(argv=None):    # pylint: disable=unused-argument
    count()

if __name__ == '__main__':
    tf.app.run()
