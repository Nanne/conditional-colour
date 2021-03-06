"""Converts data to TFRecords file format with Example protos.
images are resized to have the shortest side be 256.

Requires the rijksgt.json as generated by matlabeler.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, pandas, random
import json
from random import shuffle, seed
import string
import tensorflow as tf
import numpy as np

from scipy import misc
from scipy.misc import imread, imresize
from skimage import color
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')

tf.app.flags.DEFINE_string('images_root', '/data/wikiart/allimg',
                           'Location where original images are stored')
tf.app.flags.DEFINE_string('directory', '/data/wikiart/tfrecords',
                           'Directory to write the converted result to')
tf.app.flags.DEFINE_string('input_csv', 'all_data_info.csv',
                           'Input CSV to process')
tf.app.flags.DEFINE_integer('min_count', 5,
                            """Minimal number of artworks per artist""")

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def write_to(writer, filename, artist, ix, shortest_side=256, LAB=True):
    image_raw = misc.imread(filename)

    height = image_raw.shape[0] 
    width = image_raw.shape[1]

    if len(image_raw.shape) < 3:
        image_raw = np.tile(np.reshape(image_raw, [height, width, 1]), [1,1,3])

    if height > shortest_side or width > shortest_side:
        if height < width:
            new_height = shortest_side
            new_width = round(width * (new_height/height))
        else:
            new_width = shortest_side
            new_height = round(height * (new_width/width))
        image_raw = misc.imresize(image_raw, (int(new_height), int(new_width)), interp='bicubic')
    else:
        new_height, new_width = height, width

    if LAB: 
        image_raw = color.rgb2lab(image_raw.astype(np.float32) / 255.0) / 100

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(int(new_height)),
        'width': _int64_feature(int(new_width)),
        'depth': _int64_feature(int(image_raw.shape[2])),
        'label': _int64_feature(int(artist)),
        'id': _int64_feature(int(ix)),
        'image_raw': _bytes_feature(image_raw.tostring())}))
    writer.write(example.SerializeToString())

def main(argv):
    random.seed(731)

    df = pandas.read_csv(open(FLAGS.input_csv, 'r'))
    del df['date']
    imgs = []
    uartists = Counter()

    for index, row in df.iterrows():
        if row['title'] != row['title']:
            row['title'] = ''
        row['id'] = index
        imgs.append(row.to_dict())

        uartists[row['artist']] += 1

    for artist, count in uartists.most_common():
        if count < FLAGS.min_count: 
            uartists[artist] = -1
    
    uartists += Counter() # delete ones with negative count

    a2i,i2a = {}, {}
    for ua in list(uartists):
        a2i[ua] = len(a2i)
        i2a[a2i[ua]] = ua

    mappings = {}
    mappings['a2i'] = a2i
    mappings['i2a'] = i2a

    with open('mappings.json', 'w') as f:
        json.dump(mappings, f)

    print('Artists: %d' % (len(list(uartists))))
    
    shuffle(imgs) #randomize order

    train_rec = os.path.join(FLAGS.directory, 'train.tfrecords')
    val_rec = os.path.join(FLAGS.directory, 'validation.tfrecords')
    test_rec = os.path.join(FLAGS.directory, 'test.tfrecords')

    """writers = {}
    writers['train'] = tf.python_io.TFRecordWriter(train_rec)
    writers['val'] = tf.python_io.TFRecordWriter(val_rec)
    writers['test'] = tf.python_io.TFRecordWriter(test_rec)"""

    # how many do we want of each?
    valimg = 5000
    testimg = 5000
    # train is remainder
    N = len(imgs)

    X = []
    y = []
    for i,img in enumerate(imgs):
        if img['artist'] in a2i.keys():
            X.append(img['id'])
            y.append(a2i[img['artist']])

    # create splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testimg, random_state=731, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=valimg, random_state=731, stratify=y_train)

    # Count to make sure
    counts = {}
    counts['train'] = 0 
    counts['val'] = 0 
    counts['test'] = 0 

    metadata = {}
    metadata['train'] = []
    metadata['val'] = []
    metadata['test'] = []

    for i,img in enumerate(imgs):
        fn = os.path.join(FLAGS.images_root, img['new_filename'])

        if img['id'] in X_train:
            dset = 'train' 
        elif img['id'] in X_test:
            dset = 'test' 
        elif img['id'] in X_val:
            dset = 'val' 
        else:
            # This instance was deleted due to the artist having too few occurances
            continue

        counts[dset] += 1

        meta_tmp = {'filename': fn, 'artist_id' : a2i[img['artist']], 'artist': img['artist'], 'id': img['id']}
        metadata[dset].append(meta_tmp)

        """write_to(writers[dset],
                fn, 
                a2i[img['artist']],
                img['id'],
                shortest_side=256) """
        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N)) 
    print('train %d, val %d, test %d' % (counts['train'], counts['val'], counts['test']))
    metadata['counts'] = counts

    with open('metadata.json', 'w') as f:
        json.dump(metadata, f)

    for pth, splt in zip([train_rec, val_rec, test_rec], ['train', 'val', 'test']):
        with open(pth + ".json") as f:
            json.dump({"count": counts[splt]}, f)

    #for w in writers.values():
    #    w.close()

if __name__ == '__main__':
    tf.app.run()
