# EDSR Tensorflow Implementation
An implementation of [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921.pdf) written in tensorflow.

## Requirements
 - Tensorflow
 - scipy
 - tqdm
 - argparse

## Installation
 `pip install -r requirements.txt`

## Training
In order to train, you'll have to do a few things...
 - Download a dataset of images (due to my computational limitations, I've used [General-100](https://drive.google.com/file/d/0B7tU5Pj1dfCMVVdJelZqV0prWnM/view?usp=drive_web))
 - Place all the images from that dataset into a directory under this one
 - run `python train.py --dataset data_dir` where data_dir is the directory containing your images
 - In order to view stats during training (image previews, scalar for loss), simply run `tensorboard --logdir your_save_directory` where `your_save_directory`
 is the directory you passed in as the save directory argument for training (`saved_models` by default)

## Training Details
As I've mentioned before, I'm currently faced with some computational limitations, so this
caused me to do a few things differently than what is mentioned in the paper. One of the
more important changes I've made was using the General-100 dataset, because it's much smaller.
I've also trained a network with less layers than the original baseline model as was described 
in the paper. This, however, can still be done using my code by adjusting some training parameters.
I've trained by taking the center 100x100 pixels of each image in General-100, and shrinking them down to 50x50.
I then trained an EDSR to resize the 50x50 pixel images back to 100x100. Currently, I use 80% of the
dataset as a training set and 20% as a testing set. I trained the EDSR over 1000 iterations using Adam optimizer

## Using Trained Network
In order to use trained weights you just have to run this command `python test.py`. By default, this will take a random sample of
five images from your dataset, compute their output, and save it in the `out` directory. If you'd like to just run superresolution on
one image, you can run `python test.py --image your_picture` where `your_picture` is the image file you'd like to run superresolution on.

## Results
These results were computed on a network using 3 layers, and a feature size of 16. The
network was trained to scale 50x50px images to 100x100px for 1000 iterations. <br />
<br />
Updates coming soon.......
<br />


| Original image | Shrunk image | EDSR Output |
| -------------- | ------------ | ----------- |
| ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/correct0.png "Original")          | ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/input0.png "input")         | ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/output0.png "shrunk")        |
| ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/correct1.png "Original")          | ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/input1.png "input")         | ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/output1.png "shrunk")        |
| ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/correct2.png "Original")          | ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/input2.png "input")         | ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/output2.png "shrunk")        |
| ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/correct3.png "Original")          | ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/input3.png "input")         | ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/output3.png "shrunk")        |
| ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/correct4.png "Original")          | ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/input4.png "input")         | ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/output4.png "shrunk")        |

## Future Work
- Add MDSR implementation
- Train and post results on a larger model and dataset

## Pre-trained model
There is a pre-trained model that I have made for the baseline model (default params) using my machine. If you'd like to use it, download the files [here](https://drive.google.com/drive/folders/1KaotYQZb842OGHPujsijNgwdEKQo6oF9?usp=sharing). In order to run, create a directory called `saved_models` and place the files in there. Then you can use test.py to resize images.
