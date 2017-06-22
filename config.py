import tensorflow as tf
import numpy as np

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoints/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_epoch', 10,
                            """Number of epochs to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('artistprior', False,
                            """Condition prior on artist?""")
tf.app.flags.DEFINE_boolean('validate', True,
                            """Perform validation during training?""")
tf.app.flags.DEFINE_float('lr', 0.001,
                            """Initial learning rate""")
tf.app.flags.DEFINE_string('data_dir', '/exp/nanne/wikiart/tfrecords',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_integer('save_every', 1,
                            """Save every N epochs.""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of examples to process in a batch.""")
tf.app.flags.DEFINE_integer('num_categories', 1678,
                            """Number of categories for Conditional Instance Normalisation.""")
tf.app.flags.DEFINE_integer('imsize', 224,
                            """Size of input/output images""")
tf.app.flags.DEFINE_integer('quantiles', 32,
                            """Number of quantiles to split the A & B colour space in""")
tf.app.flags.DEFINE_string('network', 'CIN',
                           """CIN|IN|BN""")
tf.app.flags.DEFINE_boolean('GAN', False,
                            """Use GAN? Not really tested.""")
tf.app.flags.DEFINE_string('epoch', '-1',
                           """Choose which checkpoint file to restore, defaults to latest""")
tf.app.flags.DEFINE_string('prior_file', 'counts/counts.hdf5',
                            """Where to read the priors""")

# List of flags to restore when resuming a model
restore_flags = ['artistprior', 'num_categories', 'imsize', 
        'quantiles', 'network', 'prior_file', 'GAN']


# FIXME: make work if other value than default
# calculate centroids of the quantiles
centroids = np.array(range(0,tf.app.flags.FLAGS.quantiles)) / (tf.app.flags.FLAGS.quantiles/2.0)-1 + (1/tf.app.flags.FLAGS.quantiles)
