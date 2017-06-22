from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
from math import ceil

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

import dataprovider, util, config
import CNN.ops as ops
import CNN.GAN as GAN

import json 

EPS = 1e-12

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('resume', False,
                            """Resume training?""")
tf.app.flags.DEFINE_boolean('cat_change', False,
                            """Did the number of categories change?""")
tf.app.flags.DEFINE_string('resume_dir', '',
                           """Directory where to read checkpoints """
                           """to resume from.""")

FLAGS.network = FLAGS.network.upper()
if FLAGS.network == 'CIN':
    import CNN.colour_unet_cin as CNN
elif FLAGS.network == 'IN':
    import CNN.colour_unet_in as CNN
elif FLAGS.network == 'BN':
    import CNN.colour_unet_bn as CNN
else:
    raise ValueError('Unknown network option')

def get_train_op(total_loss, global_step, var_list=None, lr_modifier=1.0):
    """Train model.
    Create an optimizer and apply to all trainable variables. 

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
        train_op: op for training.
    """

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.tuple(update_ops)
        total_loss = control_flow_ops.with_dependencies(updates, total_loss)

    # Compute gradients.
    train_op = tf.train.AdamOptimizer(FLAGS.lr*lr_modifier).minimize(total_loss, 
            var_list=var_list, global_step=global_step)

    return train_op

def loss(pred, target, prior = [], labels = []):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
    pred: [N, height, width, 2] from inference().
    target: [N, height, width, 2] from data in -1 to 1 range.
    prior: [K,2] priors per bin for A and B

    Returns:
    Loss tensor of type float.
    """

    target_a = tf.reshape(target[:,:,:,0], (-1,))
    target_b = tf.reshape(target[:,:,:,1], (-1,))
    a,b = tf.split(axis=3, num_or_size_splits=2, value=pred)
    pred_a = tf.reshape(a, (-1, FLAGS.quantiles))
    pred_b = tf.reshape(b, (-1, FLAGS.quantiles))

    # quantize by rescaling and casting to int:
    q_target_a = tf.to_int32((target_a+1) / (2/(FLAGS.quantiles-1)))
    q_target_b = tf.to_int32((target_b+1) / (2/(FLAGS.quantiles-1)))

    loss_a = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_a, labels=q_target_a) 
    loss_b = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_b, labels=q_target_b) 

    if len(prior) > 0:
        if len(prior.shape) == 2:
            prior_A = tf.constant(prior[:,0])
            prior_B = tf.constant(prior[:,1])

            loss_a = tf.multiply(loss_a, tf.gather(prior_A, q_target_a))
            loss_b = tf.multiply(loss_b, tf.gather(prior_B, q_target_b))
        else:
            prior_A = tf.constant(prior[:, :, 0])
            prior_B = tf.constant(prior[:, :, 1])

            # get one artist label for each pixel
            labels = tf.reshape(labels, (labels.get_shape()[0].value, 1, 1))
            labels = tf.tile(labels, (1, target.get_shape()[1].value, target.get_shape()[2].value))
            labels = tf.to_int32(tf.reshape(labels, (-1,)))

            loss_a = tf.multiply(loss_a, tf.gather_nd(prior_A, tf.stack((labels, q_target_a), axis=1)))
            loss_b = tf.multiply(loss_b, tf.gather_nd(prior_B, tf.stack((labels, q_target_b), axis=1)))

    loss = tf.reduce_mean(loss_a) + tf.reduce_mean(loss_b)

    tf.add_to_collection('losses', loss)

    f_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    q_a = tf.one_hot(tf.reshape(q_target_a, target[:,:,:,0].get_shape()), depth=FLAGS.quantiles)
    q_b = tf.one_hot(tf.reshape(q_target_b, target[:,:,:,1].get_shape()), depth=FLAGS.quantiles)

    quanAB = tf.concat((q_a, q_b), axis=3)

    return f_loss, quanAB

def train():
    """Train CNN until max steps have been reached."""
    with tf.Graph().as_default():
        tf.set_random_seed(1234)

        global_step = tf.contrib.framework.get_or_create_global_step()

        L, AB, labels, N_train = dataprovider.distorted_inputs(get_size=True)
        if FLAGS.validate:
            vL, vAB, vlabels, N_val = dataprovider.inputs(get_size=True)

        isTrain = tf.placeholder_with_default(True, (), name='istrain')
        keep_prob = tf.placeholder_with_default(1.0, (), name='keep_prob')

        # make templates so we can share variables between train/val
        inference = tf.make_template('inference', 
                CNN.inference, 
                is_training=isTrain, 
                keep_prob=keep_prob)

        if not FLAGS.artistprior:
            pf = util.get_prior(FLAGS.prior_file)
        else:
            pf = util.get_conditionals(FLAGS.prior_file)

        predicted = inference(L, labels)
        trloss, quanAB = loss(predicted, AB, pf, labels)
        tf.summary.scalar('softmax loss', trloss)
        
        if FLAGS.validate:
            vpredicted = inference(vL, vlabels)
            vloss, _ = loss(vpredicted, vAB, pf, vlabels)
            tf.summary.scalar('validation_loss', vloss)
            
        if FLAGS.GAN == True:
            # create two copies of discriminator,
            # one for real pairs and one for fake pairs
            # they share the same underlying variables
            with tf.name_scope("real_discriminator"):
                with tf.variable_scope("discriminator"):
                        predict_real = GAN.discriminator(L, quanAB, 64)

            with tf.name_scope("fake_discriminator"):
                with tf.variable_scope("discriminator", reuse=True):
                    predict_fake = GAN.discriminator(L, predicted, 64)

            with tf.name_scope("discriminator_loss"):
                # minimizing -tf.log will try to get inputs to 1
                # predict_real => 1
                # predict_fake => 0
                discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

            with tf.name_scope("generator_gan_loss"):
                gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))

            tf.summary.scalar('discriminator loss loss', discrim_loss)
            tf.summary.scalar('GAN loss', gen_loss_GAN)

            total_loss = trloss * 0.999 + gen_loss_GAN * 0.001
        else:
            total_loss = trloss

        train_op = get_train_op(total_loss, global_step)

        #TODO FIX THIS, there must be a more elegant way of doing this
        if FLAGS.cat_change:
            var_list = []
            init_list = []
            for vr in tf.global_variables():
                if ('gamma' + str(FLAGS.num_categories) in vr.name or 
                    'beta' + str(FLAGS.num_categories) in vr.name):
                    init_list.append(vr)
                else:
                    var_list.append(vr)

            tf.variables_initializer(init_list)
            # overwrite trainop
            train_op = get_train_op(trloss, global_step, init_list, 1)
            #o_train_op = get_train_op(trloss, global_step, var_list, 0.001)
            #train_op = tf.group(i_train_op, o_train_op)
            saver = tf.train.Saver(var_list, 
                max_to_keep=None)
        else:
            saver = tf.train.Saver(tf.global_variables(), 
                max_to_keep=None)

        summary_op = tf.summary.merge_all()

        gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC')
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement, 
            gpu_options=gpu_options, allow_soft_placement=True))

        if FLAGS.resume: 
            if len(FLAGS.resume_dir) > 0:
                ckpt = tf.train.get_checkpoint_state(FLAGS.resume_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    if FLAGS.epoch == '-1':
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        ckpt_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    else:
                        restore_path = ''
                        for pth in ckpt.all_model_checkpoint_paths:
                            if pth.split('-')[-1] == FLAGS.epoch:
                                restore_path = pth
                                break
                        if restore_path == '':
                            print('No checkpoint found matching epoch:', FLAGS.epoch)
                            return

                        saver.restore(sess, restore_path)
                        ckpt_step = restore_path.split('/')[-1].split('-')[-1]

                    print('Restoring checkpoint from iteration:', ckpt_step)
                else:
                    print('No checkpoint found')
                    return
            else:
                raise NotImplementedError("Resuming from latest not implemented yet")

        if FLAGS.cat_change:
            # Given that this is the finetuned version of a previous model
            # we'll assume for now we don't need to really finetune this model
            # so only save the trainable variables, and not any gradient info
            saver = tf.train.Saver(tf.trainable_variables(), 
                max_to_keep=None)
            # Init variables which weren't restored from saved
            variables = tf.global_variables()
            init_flag = sess.run(
                tf.stack([tf.is_variable_initialized(v) for v in variables]))
            
            init = tf.variables_initializer([v for v, f in zip(variables, init_flag) if not f])
        else:
            init = tf.global_variables_initializer()
        
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir, sess.graph)

        avg_loss = []
        steps_per_epoch = int(N_train / FLAGS.batch_size)
        save_step = steps_per_epoch * FLAGS.save_every
        disp_step = max(5, int(steps_per_epoch / 100)) # 100 times per epoch
        max_steps = steps_per_epoch * FLAGS.max_epoch
        start_time = time.time()
        correct_pred = []

        for step in xrange(0, max_steps+1):
            _, loss_value, gstep = sess.run([train_op, total_loss, global_step], feed_dict={keep_prob:0.5})
            avg_loss.append(loss_value)

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN ' + str(avg_loss) 

            if gstep % disp_step == 0 or gstep == 0:

                duration = time.time() - start_time
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = (num_examples_per_step*min(gstep, disp_step)) / duration
                sec_per_batch = float(duration) / min(step, disp_step)

                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), gstep, np.mean(np.array(avg_loss)), 
                                     examples_per_sec, sec_per_batch))
                avg_loss = []
                start_time = time.time()

            # Save the model checkpoint periodically.
            if gstep % save_step == 0 or (step + 1) == max_steps:
                epoch = int(gstep/steps_per_epoch)
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)

            if gstep % steps_per_epoch == 0 and gstep > 0:
                epoch = int(gstep/steps_per_epoch)
                if FLAGS.validate:
                    validation_loss = 0
                    val_steps = max(1,int(N_val/FLAGS.batch_size))
                    for v_step in xrange(val_steps):  
                        loss_value = sess.run(vloss, feed_dict={isTrain:False})
                        validation_loss += loss_value
                    print ('%s: epoch %d, validation loss = %.4f' % (datetime.now(), 
                       epoch, validation_loss / val_steps))
                    start_time = time.time() # reset this otherwise it looks like train has gotten slow.
                else:
                    print ('%s: epoch %d' % (datetime.now(), epoch))

            if gstep % disp_step == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, gstep)

def main(argv=None):  # pylint: disable=unused-argument
    if not (FLAGS.resume and FLAGS.checkpoint_dir == FLAGS.resume_dir):
        if tf.gfile.Exists(FLAGS.checkpoint_dir):
            raise NameError('Training directory already exists!')
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    with open(os.path.join(FLAGS.checkpoint_dir, 'flags.json'), 'w') as f:
        f.write(json.dumps(FLAGS.__dict__['__flags']))

    train()

if __name__ == '__main__':
    tf.app.run()
