# EDSR Tensorflow Implementation
An implementation of [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921.pdf) written in tensorflow.

##Requirements
 - Tensorflow
 - scipy
 - tqdm

##Installation
 `pip install -r requirements.txt`

##Training
In order to train, you'll have to do a few things...
 - Download a dataset of images (due to my computational limitations, I've used General-100)
 - (If you'd like to use General-100, run ./download_data.sh)
 - Place all the images from that dataset into a directory under this one
 - run `python train.py --dataset data_dir` where data_dir is the directory containing your images

##Training details
As I've mentioned before, I'm currently faced with some computational limitations, so this
caused me to do a few things differently than what is mentioned in the paper. One of the
more important changes I've made was using the General-100 dataset, because it's much smaller.
I've also trained a network with less layers than the original baseline model as was described 
in the paper. This, however, can still be done using my code by adjusting some training parameters.
I've trained by taking the center 100x100 pixels of each image in General-100, and shrinking them down to 50x50.
I then trained an EDSR to resize the 50x50 pixel images back to 100x100. Currently, I use 80% of the
dataset as a training set and 20% as a testing set. I trained the EDSR over 1000 iterations using Adam optimizer

##Results
| Original image | Shrunk image | EDSR Output |
| -------------- | ------------ | ----------- |
| TODO           | TODO         | TODO        |

##Remarks
TODO

##Future work
- Add MDSR implementation
- Train and post results on a larger model and dataset
