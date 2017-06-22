# Artist specific colourisation
Tensorflow implementation for paper "A learned representation of artist specific colourisation"

## Examples

Given a greyscale image we would like to produce an as realistic as possible colour image, in the style of the artist.
Some examples using Conditional Instance Normalisation:

<table>
<tr>
<th>Original</th><th>Greyscale input</th><th>Colourisation</th>
</tr>
<tr>
<td><img src="/examples/29_3738_original.png"/></td>
<td><img src="/examples/29_3738_grey.png"/></td>
<td><img src="/examples/29_3738_predicted.png"/></td>
</tr>
<tr>
<td><img src="/examples/1214_1362_original.png"/></td>
<td><img src="/examples/1214_1362_grey.png"/></td>
<td><img src="/examples/1214_1362_predicted.png"/></td>
</tr>
</table>

## Setup

Dependencies
* [Tensorflow](https://www.tensorflow.org/install/)
* [Scipy](https://www.scipy.org/install.html)
* [h5py](http://docs.h5py.org/en/latest/build.html)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cudnn](https://developer.nvidia.com/cudnn)

Dataset
[Painters by numbers](https://www.kaggle.com/c/painter-by-numbers)

## Training

* Run data/make_tf_records.py on image data to create tfrecords
* Run count.py to gather statistics for re-weighting of colours
* Run train.py

## Test

Run eval.py with correct --source
