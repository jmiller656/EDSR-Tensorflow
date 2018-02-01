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
def resBlock(x,channels=64,kernel_size=[3,3],scale=1):
	tmp = slim.conv2d(x,channels,kernel_size,activation_fn=None)
	tmp = tf.nn.relu(tmp)
	tmp = slim.conv2d(tmp,channels,kernel_size,activation_fn=None)
	tmp *= scale
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
	x = slim.conv2d(x,features,[3,3],activation_fn=activation)
	if scale == 2:
		ps_features = 3*(scale**2)
		x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
		#x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
		x = PS(x,2,color=True)
	elif scale == 3:
		ps_features =3*(scale**2)
		x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
		#x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
		x = PS(x,3,color=True)
	elif scale == 4:
		ps_features = 3*(2**2)
		for i in range(2):
			x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
			#x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
			x = PS(x,2,color=True)
	return x

"""
Borrowed from https://github.com/tetrachrome/subpixel
Used for subpixel phase shifting after deconv operations
"""
def _phase_shift(I, r):
	bsize, a, b, c = I.get_shape().as_list()
	bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
	X = tf.reshape(I, (bsize, a, b, r, r))
	X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
	X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
	X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, b, a*r, r
	X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
	X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, a*r, b*r
	return tf.reshape(X, (bsize, a*r, b*r, 1))

"""
Borrowed from https://github.com/tetrachrome/subpixel
Used for subpixel phase shifting after deconv operations
"""
def PS(X, r, color=False):
	if color:
		Xc = tf.split(X, 3, 3)
		X = tf.concat([_phase_shift(x, r) for x in Xc],3)
	else:
		X = _phase_shift(X, r)
	return X

"""
Tensorflow log base 10.
Found here: https://github.com/tensorflow/tensorflow/issues/1666
"""
def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator
