import h5py, json
from scipy.ndimage.filters import gaussian_filter1d
import tensorflow as tf
import numpy as np
import os.path
import math

import config
FLAGS = tf.app.flags.FLAGS

ix2artist = json.load(open('dataset/mappings.json', 'r'))['i2a']

def ix2artist(ix): # convenience function to look up artists name
    return ix2artist[str(ix)]

def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out

def psnr(mse, PIXEL_MAX = 255.0):
    if mse == 0:
        return 100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def get_conditionals(fname = 'counts/counts.hdf5', sigma=1, gamma=0.10, alpha=1):
    # procedure based on https://arxiv.org/abs/1603.08511

    prior = np.zeros((1584, 32, 2), dtype=np.float32)

    uniform = np.ones((32,))
    uniform /= np.sum(uniform)

    with h5py.File(fname, "r") as f:
        labels = np.array(f['labels'])
        artists = np.unique(labels)

        for i,artist in enumerate(artists):
            probs_A = gaussian_filter1d(np.sum(f['Acounts'][(labels == artist).squeeze(),:], 0), sigma, mode='constant')
            probs_A /= np.sum(probs_A)
            probs_B = gaussian_filter1d(np.sum(f['Bcounts'][(labels == artist).squeeze(),:], 0), sigma, mode='constant')
            probs_B /= np.sum(probs_B)

            prior_mix_A =  ((1-gamma)*probs_A + gamma*uniform)
            prior_mix_B =  ((1-gamma)*probs_B + gamma*uniform)

            prior_factor_A = prior_mix_A**-alpha
            prior[i, :,0] = prior_factor_A/np.sum(probs_A*prior_factor_A)

            prior_factor_B = prior_mix_B**-alpha
            prior[i, :,1] = prior_factor_B/np.sum(probs_B*prior_factor_B)

    return prior

def get_prior(fname = 'counts/counts.hdf5', sigma=5, gamma=0.5, alpha=1):
    # procedure based on https://arxiv.org/abs/1603.08511

    prior = np.zeros((FLAGS.quantiles, 2), dtype=np.float32)

    uniform = np.ones((FLAGS.quantiles,))
    uniform /= np.sum(uniform)

    with h5py.File(fname, "r") as f:
        probs_A = gaussian_filter1d(np.sum(f['Acounts'], 0), sigma, mode='constant')
        probs_A /= np.sum(probs_A)
        probs_B = gaussian_filter1d(np.sum(f['Bcounts'], 0), sigma, mode='constant')
        probs_B /= np.sum(probs_B)

        prior_mix_A =  ((1-gamma)*probs_A + gamma*uniform)
        prior_mix_B =  ((1-gamma)*probs_B + gamma*uniform)

        prior_factor_A = prior_mix_A**-alpha
        prior[:,0] = prior_factor_A/np.sum(probs_A*prior_factor_A)

        prior_factor_B = prior_mix_B**-alpha
        prior[:,1] = prior_factor_B/np.sum(probs_B*prior_factor_B)

    return prior


def restore_flags():
    if tf.gfile.Exists(os.path.join(tf.app.flags.FLAGS.checkpoint_dir, 'flags.json')):
        with open(os.path.join(tf.app.flags.FLAGS.checkpoint_dir, 'flags.json'), 'r') as f:
            print('Restoring training flags')
            train_flags = json.load(f)

            for key in config.restore_flags:
                if key in train_flags:
                    tf.app.flags.FLAGS.__dict__['__flags'][key] = train_flags[key]
                print(key, tf.app.flags.FLAGS.__dict__['__flags'][key])
    else:
        print('No flag configuration file found, using default flags')
    return
