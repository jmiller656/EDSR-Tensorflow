import tensorflow as tf
import tensorflow.contrib.slim as slim

"""
Creates a convolutional residual block
as defined in the paper. More on
this inside model.py

x: input to pass through the residual block
channels: number of channels to compute
stride: convolution stride
"""
def resBlock(x,channels=64,stride=[3,3]):
	tmp = slim.conv2d(x,channels,stride,activation_fn=None)
	tmp = tf.nn.relu(tmp)
	tmp = slim.conv2d(tmp,channels,stride,activation_fn=None)
	return x + tmp

"""
Method to upscale an image using
conv2d transpose. Based on upscaling
method defined in the paper

x: input to be upscaled
scale: scale increase of upsample
features: number of features to compute
activation: activation function
"""
def upsample(x,scale=2,features=64,activation=tf.nn.relu):
	assert scale in [2,3,4]
	if scale == 2:
		x = slim.conv2d_transpose(x,features,6,stride=2,activation_fn=activation)
	elif scale == 3:
		x = slim.conv2d_transpose(x,features,9,stride=3,activation_fn=activation)
	elif scale == 4:
		for i in range(2):
			x = slim.conv2d_transpose(x,features,6,stride=2,activation_fn=activation)
	return x


