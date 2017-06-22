from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib 
matplotlib.use('Agg') 

from datetime import datetime
import os.path, os, sys
import glob, math
import json
from scipy import misc
from skimage import color

import numpy as np

from skimage import color
from scipy.misc import imread
import scipy.ndimage.interpolation as sni

def psnr(mse, PIXEL_MAX = 255.0):
    if mse == 0:
        return 100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

source = 're-eval'
gt_dir = 'visuals/cp_instancenorm/'
pred_model = sys.argv[1].strip()

if not os.path.exists(pred_model):
    raise ValueError('No such model dir: '+ pred_model)

pred_dir = os.path.join('visuals/', pred_model) + "/"

if not os.path.exists(pred_dir):
    raise ValueError('No visuals found for: '+ pred_dir)

gtlist = sorted([fn for fn in glob.glob(gt_dir + "*_original.png")])
num_examples = len(gtlist)

predlist = sorted([fn for fn in glob.glob(pred_dir + "*_predicted.png")])
assert num_examples == len(predlist), "Unequal number of gt (%d) and pred (%d)." % (num_examples, len(predlist))

def benchmark():
    """Do exact same evaluation for all methods, as colour conversions introduce some noise."""

    # per file metrics
    psnr_scores = np.zeros((num_examples, ))
    square_error = np.zeros((num_examples, ))
    baseline_rmse = np.zeros((num_examples, ))
    baseline_psnr = np.zeros((num_examples, ))
    pixels = np.zeros((num_examples, ))
    artists = np.zeros((num_examples, ))

    for s,(gt_fn,pred_fn) in enumerate(zip(gtlist, predlist)):
        artist = int(gt_fn.split('/')[-1].split('_')[0])
        step = int(gt_fn.split('/')[-1].split('_')[1])

        gt_raw = misc.imread(gt_fn).astype(np.float64)
        gt_lab = (color.rgb2lab(gt_raw / 255.0) / 100)
        gtL = gt_lab[:, :, 0].reshape((224,224,1))
        gtAB = gt_lab[:, :, 1:3]

        pred_raw = misc.imread(pred_fn).astype(np.float64)
        pred_lab = (color.rgb2lab(pred_raw / 255.0) / 100)
        predL = pred_lab[:, :, 0].reshape((224,224,1))
        predAB = pred_lab[:, :, 1:3]

        psnr_scores[step] = psnr(np.mean(np.square(gt_raw-pred_raw)),255)

        base = np.zeros_like(gtAB)
        base_rgb = color.lab2rgb(np.concatenate((gtL, base), axis=2)*100)*255.0

        pixels[step] = np.prod(gtAB.shape)
        baseline_rmse[step] = np.sum(np.square(base-gtAB))
        baseline_psnr[step] = psnr(np.mean(np.square(gt_raw-base_rgb)),255)
        square_error[step] = np.sum(np.square(predAB-gtAB))
        artists[step] = artist

        if s % 100 == 0:
            print('processing %d/%d (%.2f%% done)' % (s, num_examples, s*100.0/num_examples))

    N = np.sum(pixels)
    RMSE_method = np.sqrt(np.sum(square_error) / N)
    RMSE_baseline = np.sqrt(np.sum(baseline_rmse) / N)
    PSNR_baseline = np.mean(baseline_psnr)
    PSNR_method = np.mean(psnr_scores)
    print('%s: %d pixels and %d artworks\nmethod RMSE = %.4f' % (
        datetime.now(), N, num_examples, RMSE_method))
    print('method PSNR = %.4f' % (PSNR_method))
    print('baseline RMSE = %.4f' % (RMSE_baseline))
    print('baseline PSNR = %.4f' % (PSNR_baseline))
    
    if not os.path.exists(os.path.join(pred_model, 'results/')):
        os.makedirs(os.path.join(pred_model, 'results/'))

    summary_file = os.path.join(pred_model, 'results/', 'summary.json')
    if os.path.exists(summary_file):
        summary = json.load(open(summary_file, 'r'))
    else:
        summary = {}

    with open(summary_file, 'w') as r: 
        if source not in summary.keys():
            summary[source] = {}

        summary[source]['baseline'] = {'RMSE' : RMSE_baseline, 'PSNR' : PSNR_baseline}
        summary[source]['method'] = {'RMSE': RMSE_method, 'PSNR': PSNR_method}

        r.write(json.dumps(summary))

    result_fn = 'reeval_results_' + source + '.json'
    with open(os.path.join(pred_model, 'results/', result_fn), 'w') as r: 
        result_ = {'summary' : 
                {'pixels': N, 'RMSE': RMSE_method, 'PSNR': PSNR_method},
                    'per_file' : [], 'datetime': str(datetime.now())}

        result_['per_file'] = {'artists': artists.tolist(), 
                                'PSNR': psnr_scores.tolist(),
                                'pixels': pixels.tolist(), 
                                'SE': square_error.tolist(), 
                                'baseline_psnr': PSNR_baseline.tolist(),
                                'baseline_rmse': RMSE_baseline.tolist() }

        r.write(json.dumps(result_))

def main(argv=None):    # pylint: disable=unused-argument
    benchmark()

if __name__ == '__main__':
    main()
