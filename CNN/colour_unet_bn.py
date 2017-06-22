from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import CNN.ops as ops
import config, util
FLAGS = tf.app.flags.FLAGS

def inference(images, labels=[], is_training=None, keep_prob=tf.constant(1.0)):
    """Build the model.

    Args:
        images: Images to train on, placeholder or pipeline.

    Returns:
        Image embedding.
    """
    
    # input is 224x224
    conv1 = ops.conv2d('conv1', images, [4,4,1,64], stride=[1,2,2,1]) # 112x112
    conv2 = ops.conv2d_bn('conv2', conv1, [4,4,64,128], is_training, stride=[1,2,2,1]) # 56x56
    conv3 = ops.conv2d_bn('conv3', conv2, [4,4,128,256], is_training, stride=[1,2,2,1]) # 28x28
    conv4 = ops.conv2d_bn('conv4', conv3, [4,4,256,512], is_training, stride=[1,2,2,1]) # 14x14
    conv5 = ops.conv2d_bn('conv5', conv4, [4,4,512,512], is_training, stride=[1,2,2,1]) # 7x7
    conv6 = ops.conv2d_bn('conv6', conv5, [4,4,512,512], is_training, stride=[1,2,2,1]) # 4x4
    conv7 = ops.conv2d_bn('conv7', conv6, [4,4,512,512], is_training, stride=[1,2,2,1]) # 2x2
    conv8 = ops.conv2d_bn('conv8', conv7, [4,4,512,512], is_training, stride=[1,2,2,1]) # 1x1

    # upsample
    up1_1 = tf.image.resize_nearest_neighbor(conv8, [2,2], name='up1') # 2x2
    up1 = ops.conv2d_bn('upc1', up1_1, [4,4,512,512], is_training)
    up1 = tf.nn.dropout(up1, keep_prob)
    up1 = tf.concat(axis=3, values=[up1, conv7])

    up2_1 = tf.image.resize_nearest_neighbor(up1, [4,4], name='up2') # 4x4
    up2 = ops.conv2d_bn('upc2', up2_1, [4,4,1024,512], is_training)
    up2 = tf.nn.dropout(up2, keep_prob)
    up2 = tf.concat(axis=3, values=[up2, conv6])

    up3_1 = tf.image.resize_nearest_neighbor(up2, [7,7], name='up3') # 7x7
    up3 = ops.conv2d_bn('upc3', up3_1, [4,4,1024,512], is_training)
    up2 = tf.nn.dropout(up2, keep_prob)
    up3 = tf.concat(axis=3, values=[up3, conv5])

    up4_1 = tf.image.resize_nearest_neighbor(up3, [14,14], name='up4') # 14x14
    up4 = ops.conv2d_bn('upc4', up4_1, [4,4,1024,512], is_training)
    up4 = tf.concat(axis=3, values=[up4, conv4])

    up5_1 = tf.image.resize_nearest_neighbor(up4, [28,28], name='up5') # 28x28
    up5 = ops.conv2d_bn('upc5', up5_1, [4,4,1024,256], is_training)
    up5 = tf.concat(axis=3, values=[up5, conv3])

    up6_1 = tf.image.resize_nearest_neighbor(up5, [56,56], name='up6') # 56x56
    up6 = ops.conv2d_bn('upc6', up6_1, [4,4,512,128], is_training)
    up6 = tf.concat(axis=3, values=[up6, conv2])

    up7_1 = tf.image.resize_nearest_neighbor(up6, [112,112], name='up7') # 112x112
    up7 = ops.conv2d_bn('upc7', up7_1, [4,4,256,64], is_training)
    up7 = tf.concat(axis=3, values=[up7, conv1])

    up8_1 = tf.image.resize_nearest_neighbor(up7, [224,224], name='up8') # 112x112
    up8 = ops.conv2d_bn('upc8', up8_1, [4,4,128,FLAGS.quantiles*2], is_training, activation=tf.identity)

    return up8
