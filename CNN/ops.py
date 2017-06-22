from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import re, math

def variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    var = tf.get_variable(name, shape, initializer=initializer)
    return var

def variable_with_weight_decay(name, shape, wd, init=tf.contrib.layers.xavier_initializer_conv2d()):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    var = variable_on_cpu(name, shape, init)

    # FIXME: Weight decay sometimes produces NaN loss
    # TODO: figure out what's going on and solve
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        weight_decay = tf.check_numerics(weight_decay, "weight_decay " + name + " contains non-numeric")
        tf.add_to_collection('losses', weight_decay)

    return var

def conv2d(name, bottom, shape, stride=[1,1,1,1], padding='SAME', 
        activation=tf.nn.relu, 
        initializer=tf.contrib.layers.xavier_initializer_conv2d()):
    with tf.variable_scope(name) as scope:
        kernel = variable_with_weight_decay('weights', shape=shape, wd=None, init=initializer)
        conv = tf.nn.conv2d(bottom, kernel, stride, padding=padding)
        biases = variable_on_cpu('biases', [shape[3]], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv = activation(bias, name=scope.name)
    return conv

def conv2d_bn(name, bottom, shape, isTraining, stride=[1,1,1,1], padding='SAME', 
        activation=tf.nn.relu, 
        initializer=tf.contrib.layers.xavier_initializer_conv2d(), 
        reuse=None):
    with tf.variable_scope(name) as scope:
        kernel = variable_with_weight_decay('weights', shape=shape, wd=None, init=initializer)
        conv = tf.nn.conv2d(bottom, kernel, stride, padding=padding)

        eps = 1E-5
        conv = tf.contrib.layers.batch_norm(conv, is_training=isTraining, center=True, scale=True,
                epsilon=eps,
                scope=scope, reuse=reuse, updates_collections=None)
        conv = activation(conv, name=scope.name)
    return conv

def linear(name, bottom, shape, activation=tf.identity):
    with tf.variable_scope(name) as scope:
        weights = variable_with_weight_decay('weights', shape, wd=None, init=tf.contrib.layers.xavier_initializer)
        biases = variable_on_cpu('biases', [shape[1]],
                              tf.constant_initializer(0.0))

        out = activation(tf.add(tf.matmul(bottom, weights), biases), name=scope.name)

    return out

def max_pool(name, bottom, shape=[1,2,2,1], stride=[1,2,2,1], padding='SAME'):
    pool = tf.nn.max_pool(bottom, ksize=shape, strides=stride,
                             padding=padding, name=name)
    return pool

def instance_norm(x, center=True, scale=True):
    # based conditional_instance_norm from https://github.com/tensorflow/magenta/blob/master/magenta/models/image_stylization/ops.py
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = [1,2]

    beta, gamma = None, None
    if center:
        beta = variable_with_weight_decay('beta', params_shape, wd=None, init=tf.zeros_initializer())
    if scale:
        gamma = variable_with_weight_decay('gamma', params_shape, wd=None, init=tf.ones_initializer())

    # These ops will only be performed when training.
    mean, variance = tf.nn.moments(x, axis, keep_dims=True)

    variance_epsilon = 1E-5
    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, variance_epsilon)

    return x

def conditional_instance_norm(x, labels, num_categories, center=True, scale=True):
    # based conditional_instance_norm from https://github.com/tensorflow/magenta/blob/master/magenta/models/image_stylization/ops.py
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = [1,2]

    def _label_conditioned_variable(name, initializer, labels, num_categories):
        # based on https://github.com/tensorflow/magenta/blob/master/magenta/models/image_stylization/ops.py
        """Label conditioning."""
        shape = tf.TensorShape([num_categories]).concatenate(params_shape)
        var = variable_with_weight_decay(name, shape, wd=None, init=initializer)
    
        conditioned_var = tf.gather(var, labels)
        conditioned_var = tf.expand_dims(tf.expand_dims(conditioned_var, 1), 1)
        return conditioned_var

    beta, gamma = None, None
    if center:
        beta = _label_conditioned_variable('beta' + str(num_categories), tf.zeros_initializer(), labels, num_categories)
    if scale:
        gamma = _label_conditioned_variable('gamma' + str(num_categories), tf.ones_initializer(), labels, num_categories)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis, keep_dims=True)

    variance_epsilon = 1E-5
    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, variance_epsilon)

    return x

def conv2d_in(name, bottom, shape, stride=[1,1,1,1], padding='SAME', 
        activation=tf.nn.relu, 
        initializer=tf.contrib.layers.xavier_initializer_conv2d()):
    with tf.variable_scope(name) as scope:
        kernel = variable_with_weight_decay('weights', shape=shape, wd=None, init=initializer)
        conv = tf.nn.conv2d(bottom, kernel, stride, padding=padding)

        conv = instance_norm(conv)
        conv = activation(conv, name=scope.name)

    return conv

def conv2d_cin(name, bottom, shape, labels, num_categories, stride=[1,1,1,1], padding='SAME', 
        activation=tf.nn.relu, 
        initializer=tf.contrib.layers.xavier_initializer_conv2d()):
    with tf.variable_scope(name) as scope:
        kernel = variable_with_weight_decay('weights', shape=shape, wd=None, init=initializer)
        conv = tf.nn.conv2d(bottom, kernel, stride, padding=padding)

        conv = conditional_instance_norm(conv, labels, num_categories)
        conv = activation(conv, name=scope.name)

    return conv

def lrelu(x, a):
    """
    Add leaky-relu activation.

    adding these together creates the leak part and linear part
    then cancels them out by subtracting/adding an absolute value term
    leak: a*x/2 - a*abs(x)/2
    linear: x/2 + abs(x)/2
    """
    with tf.name_scope("lrelu"):
        # this block looks like it has 2 inputs
        # on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)
