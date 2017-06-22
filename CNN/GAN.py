from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import CNN.ops as ops
import config, util
FLAGS = tf.app.flags.FLAGS

def discriminator(discrim_inputs, discrim_targets, ndf):
    """Create discriminator network."""
    n_layers = 3
    layers = []

    # 2x [batch, height, width, in_channels]
    # => [batch, height, width, in_channels * 2]
    if discrim_inputs != None:
        input = tf.concat(values=[discrim_inputs, discrim_targets], axis=3)
    else:
        input = discrim_targets

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        in_channels = input.get_shape()[3]
        convolved = ops.conv2d('conv', input, [4,4,in_channels,ndf], stride=[1,2,2,1])
        rectified = ops.lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    norm = ops.instance_norm
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            in_channels = layers[-1].get_shape()[3]
            out_channels = ndf * min(2**(i+1), 8)
            # last layer here has stride 1
            stride = 1 if i == n_layers - 1 else 2
            convolved = ops.conv2d('conv', layers[-1], [4,4,in_channels,out_channels], stride=[1,stride,stride,1])
            normalized = norm(convolved)
            rectified = ops.lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        in_channels = layers[-1].get_shape()[3]
        convolved = ops.conv2d('conv', layers[-1], [4,4,in_channels,1], stride=[1,1,1,1])
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]
