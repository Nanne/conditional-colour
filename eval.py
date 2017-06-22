from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time, math
import json
from scipy import misc
from skimage import color

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import dataprovider, util, config, train


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('colour_inference', 'expectation',
                            """How to infer colour. (mode, expectation)""")
tf.app.flags.DEFINE_boolean('store_visuals', False,
                            """Write input/target/output to disk?""")
tf.app.flags.DEFINE_string('output', 'visuals',
                            """Directory to write generated results to."""
                            """Results in: output/model_dir/*.PNG""")
tf.app.flags.DEFINE_string('source', 'val',
                            """Evaluate on training|val|test set""")
tf.app.flags.DEFINE_boolean('randomise', False,
                            """Randomize the conditional artist label""")

def eval():
    """Evaluate model and store results + visuals (latter only if specified)."""

    if FLAGS.store_visuals:
        if FLAGS.randomise:
            raise NotImplementedError('Cannot randomise and store visuals at this time')
        # Store files in FLAGS.output/FLAGS.checkpoint_dir
        # create paths if needed
        if not os.path.exists(FLAGS.output):
            os.makedirs(FLAGS.output)
        output_path = os.path.join(FLAGS.output, FLAGS.checkpoint_dir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    if FLAGS.network == 'CIN':
        import CNN.colour_unet_cin as CNN
    elif FLAGS.network == 'IN':
        import CNN.colour_unet_in as CNN
    elif FLAGS.network == 'BN':
        import CNN.colour_unet_bn as CNN
    elif FLAGS.network == 'C_IN':
        import CNN.colour_unet_c_in as CNN
    else:
        raise ValueError('Unknown network option')

    with tf.Graph().as_default() as g:
        global_step = tf.contrib.framework.get_or_create_global_step()

        if FLAGS.source.lower() not in ['val', 'train', 'test']:
            raise ValueError('Unknown data source, valid options are: val, train, test')
        L, AB, label, N_eval = dataprovider.input(FLAGS.source, get_size=True)

        if FLAGS.randomise:
            label = tf.random_uniform([1],0,FLAGS.num_categories, dtype=tf.int64, seed=0)

        isTrain = tf.placeholder_with_default(False, (), name='istrain')
        keep_prob = tf.placeholder_with_default(1.0, (), name='keep_prob')

        inference = tf.make_template('inference', CNN.inference, is_training=isTrain, keep_prob=keep_prob)

        if not FLAGS.artistprior:
            pf = util.get_prior(FLAGS.prior_file)
        else:
            pf = util.get_conditionals(FLAGS.prior_file)

        predicted = inference(L, label)
        loss, _ = train.loss(predicted, AB, pf, label)

        saver = tf.train.Saver(tf.global_variables())

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
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
                global_step = ckpt_step
            else:
                print('No checkpoint found')
                return

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord,
                                    daemon=True, start=True))

                step = 0
                if FLAGS.store_visuals:
                    # Store as ARTIST_STEP_VERSION.png
                    fn_str = output_path + '/%d_%d_%s.png'

                # per file metrics
                square_error = np.zeros((N_eval, ))
                baseline = np.zeros((N_eval, ))
                pixels = np.zeros((N_eval, ))
                psnr = np.zeros((N_eval, ))
                base_psnr = np.zeros((N_eval, ))
                losses = np.zeros((N_eval, ))
                artists = np.zeros((N_eval, ))


                while step < N_eval and not coord.should_stop():
                    gtL, gtAB, pred, loss_val, artist = sess.run([L, AB, predicted, loss, label], feed_dict={isTrain:False})

                    base = np.zeros_like(gtAB)

                    if FLAGS.colour_inference == 'mode':
                        preda = np.argmax(pred[:,:,:,:FLAGS.quantiles], 3).reshape((FLAGS.imsize,FLAGS.imsize,1))
                        predb = np.argmax(pred[:,:,:,FLAGS.quantiles:], 3).reshape((FLAGS.imsize,FLAGS.imsize,1))
                        predab = (np.concatenate((preda,predb), 2) / (FLAGS.quantiles/2)).astype('float32') - 1
                    else:
                        preda = np.dot(util.softmax(pred[:,:,:,:FLAGS.quantiles], 3), config.centroids).reshape((FLAGS.imsize,FLAGS.imsize,1))
                        predb = np.dot(util.softmax(pred[:,:,:,FLAGS.quantiles:], 3), config.centroids).reshape((FLAGS.imsize,FLAGS.imsize,1))
                        predab = np.concatenate((preda,predb), 2)

                    pixels[step] = np.prod(predab.shape)
                    baseline[step] = np.sum(np.square(base-gtAB))
                    square_error[step] = np.sum(np.square(predab-gtAB))
                    losses[step] = loss_val
                    artists[step] = artist

                    original = np.concatenate([gtL, gtAB], 3).squeeze()
                    original = color.lab2rgb(original.astype('float64')*100)

                    pred_img = np.concatenate([gtL.reshape((FLAGS.imsize,FLAGS.imsize,1)), predab], 2)
                    pred_img = color.lab2rgb(pred_img.astype('float64')*100)
                    
                    basergb = np.concatenate([gtL.reshape((FLAGS.imsize,FLAGS.imsize,1)), base.squeeze()], 2)
                    basergb = color.lab2rgb(basergb.astype('float64')*100)

                    rgb_mse = np.mean(np.square(original-pred_img))
                    base_mse = np.mean(np.square(original-basergb))

                    psnr[step] = util.psnr(rgb_mse, 1)
                    base_psnr[step] = util.psnr(base_mse, 1)

                    if FLAGS.store_visuals:
                        misc.imsave(fn_str % (artist, step, 'grey'), gtL.squeeze())
                        misc.imsave(fn_str % (artist, step, 'original'), original)
                        misc.imsave(fn_str % (artist, step, 'predicted'), pred_img)

                    step += 1
                    if step % 100 == 0:
                        print('processing %d/%d (%.2f%% done)' % (step, N_eval, 
                            step*100.0/N_eval))

                N = np.sum(pixels)
                RMSE_method = np.sqrt(np.sum(square_error) / N)
                RMSE_baseline = np.sqrt(np.sum(baseline) / N)
                avg_loss = np.mean(losses)
                avg_psnr = np.mean(psnr)
                avg_base_psnr = np.mean(base_psnr)
                print('%s: %d pixels and %d artworks\naverage loss: %.4f' % (
                    datetime.now(), N, N_eval, avg_loss))

                print('\t\t RMSE \t \t PSNR')
                print('baseline\t %.4f \t %.4f' % (RMSE_baseline, avg_base_psnr))
                print('method\t\t %.4f \t %.4f' % (RMSE_method, avg_psnr))

                if not tf.gfile.Exists(os.path.join(FLAGS.checkpoint_dir, 'results/')):
                    tf.gfile.MakeDirs(os.path.join(FLAGS.checkpoint_dir, 'results/'))

                summary_file = os.path.join(FLAGS.checkpoint_dir, 'results/', 'summary.json')
                if tf.gfile.Exists(summary_file):
                    summary = json.load(open(summary_file, 'r'))
                else:
                    summary = {}

                with open(summary_file, 'w') as r: 
                    base_metrics = {'RMSE' : RMSE_baseline, 'PSNR' : avg_base_psnr}
                    if FLAGS.source not in summary.keys():
                        summary[FLAGS.source] = {}
                        summary[FLAGS.source]['baseline'] = base_metrics
                    else: 
                        if summary[FLAGS.source]['baseline'] != base_metrics:
                            print('Stored summary baseline', summary[FLAGS.source]['baseline'], 
                                    'differs from calculated baseline', base_metrics)
                            summary[FLAGS.source]['alt_baseline'] = base_metrics

                    perf_metrics = {'RMSE': RMSE_method, 'loss': avg_loss, 'PSNR' : avg_psnr}
                    if FLAGS.randomise:
                        if 'randomised' not in summary[FLAGS.source]:
                            summary[FLAGS.source]['randomised'] = {}
                        summary[FLAGS.source]['randomised'][ckpt_step] = perf_metrics
                    else:
                        summary[FLAGS.source][ckpt_step] = perf_metrics

                    r.write(json.dumps(summary))

                if FLAGS.randomise:
                    result_fn = 'randomised_results_' + FLAGS.source + '_' + str(ckpt_step) + '.json'
                else:
                    result_fn = 'results_' + FLAGS.source + '_' + str(ckpt_step) + '.json'

                with open(os.path.join(FLAGS.checkpoint_dir, 'results/', result_fn), 'w') as r: 
                    result_ = {'summary' : 
                            {'pixels': N, 'RMSE': RMSE_method, 'PSNR': avg_psnr,
                                'baseline_rsme': RMSE_baseline, 'baseline_psnr': avg_base_psnr, 'loss': avg_loss},
                                'per_file' : [], 'datetime': str(datetime.now()), 
                                'FLAGS': FLAGS.__dict__['__flags'], 
                                'step': ckpt_step}

                    result_['per_file'] = {'artists': artists.tolist(), 
                                            'losses': losses.tolist(),
                                            'pixels': pixels.tolist(), 
                                            'SE': square_error.tolist(), 
                                            'PSNR': psnr.tolist(), 
                                            'baseline_rmse': baseline.tolist(),
                                            'baseline_psnr': base_psnr.tolist() }

                    r.write(json.dumps(result_))

            except Exception as e:    # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

    #f.close()

def main(argv=None):    # pylint: disable=unused-argument
    util.restore_flags()
    eval()

if __name__ == '__main__':
    tf.app.run()
