import tensorflow.contrib.slim as slim
import scipy.misc
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import shutil
import utils
import os

"""
An implementation of the neural network used for
super-resolution of images as described in:

`Enhanced Deep Residual Networks for Single Image Super-Resolution`
(https://arxiv.org/pdf/1707.02921.pdf)

(single scale baseline-style model)
"""
class EDSR(object):

	def __init__(self,img_size=32,num_layers=32,feature_size=256,scale=2,output_channels=3):
		print("Building EDSR...")
		self.img_size = img_size
		self.scale = scale
		self.output_channels = output_channels

		#Placeholder for image inputs
		self.input = x = tf.placeholder(tf.float32,[None,img_size,img_size,output_channels])
		#Placeholder for upscaled image ground-truth
		self.target = y = tf.placeholder(tf.float32,[None,img_size*scale,img_size*scale,output_channels])
	
		"""
		Preprocessing as mentioned in the paper, by subtracting the mean
		However, the subtract the mean of the entire dataset they use. As of
		now, I am subtracting the mean of each batch
		"""
		mean_x = 127#tf.reduce_mean(self.input)
		image_input =x- mean_x
		mean_y = 127#tf.reduce_mean(self.target)
		image_target =y- mean_y

		#One convolution before res blocks and to convert to required feature depth
		x = slim.conv2d(image_input,feature_size,[3,3])
	
		#Store the output of the first convolution to add later
		conv_1 = x	

		"""
		This creates `num_layers` number of resBlocks
		a resBlock is defined in the paper as
		(excuse the ugly ASCII graph)
		x
		|\
		| \
		|  conv2d
		|  relu
		|  conv2d
		| /
		|/
		+ (addition here)
		|
		result
		"""

		"""
		Doing scaling here as mentioned in the paper:

		`we found that increasing the number of feature
		maps above a certain level would make the training procedure
		numerically unstable. A similar phenomenon was
		reported by Szegedy et al. We resolve this issue by
		adopting the residual scaling with factor 0.1. In each
		residual block, constant scaling layers are placed after the
		last convolution layers. These modules stabilize the training
		procedure greatly when using a large number of filters.
		In the test phase, this layer can be integrated into the previous
		convolution layer for the computational efficiency.'

		"""
		scaling_factor = 0.1
		
		#Add the residual blocks to the model
		for i in range(num_layers):
			x = utils.resBlock(x,feature_size,scale=scaling_factor)

		#One more convolution, and then we add the output of our first conv layer
		x = slim.conv2d(x,feature_size,[3,3])
		x += conv_1
		
		#Upsample output of the convolution		
		x = utils.upsample(x,scale,feature_size,None)

		#One final convolution on the upsampling output
		output = x#slim.conv2d(x,output_channels,[3,3])
		self.out = tf.clip_by_value(output+mean_x,0.0,255.0)

		self.loss = loss = tf.reduce_mean(tf.losses.absolute_difference(image_target,output))
	
		#Calculating Peak Signal-to-noise-ratio
		#Using equations from here: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
		mse = tf.reduce_mean(tf.squared_difference(image_target,output))	
		PSNR = tf.constant(255**2,dtype=tf.float32)/mse
		PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)
	
		#Scalar to keep track for loss
		tf.summary.scalar("loss",self.loss)
		tf.summary.scalar("PSNR",PSNR)
		#Image summaries for input, target, and output
		tf.summary.image("input_image",tf.cast(self.input,tf.uint8))
		tf.summary.image("target_image",tf.cast(self.target,tf.uint8))
		tf.summary.image("output_image",tf.cast(self.out,tf.uint8))
		
		#Tensorflow graph setup... session, saver, etc.
		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		print("Done building!")
	
	"""
	Save the current state of the network to file
	"""
	def save(self,savedir='saved_models'):
		print("Saving...")
		self.saver.save(self.sess,savedir+"/model")
		print("Saved!")
		
	"""
	Resume network from previously saved weights
	"""
	def resume(self,savedir='saved_models'):
		print("Restoring...")
		self.saver.restore(self.sess,tf.train.latest_checkpoint(savedir))
		print("Restored!")	

	"""
	Compute the output of this network given a specific input

	x: either one of these things:
		1. A numpy array of shape [image_width,image_height,3]
		2. A numpy array of shape [n,input_size,input_size,3]

	return: 	For the first case, we go over the entire image and run super-resolution over windows of the image
			that are of size [input_size,input_size,3]. We then stitch the output of these back together into the
			new super-resolution image and return that

	return  	For the second case, we return a numpy array of shape [n,input_size*scale,input_size*scale,3]
	"""
	def predict(self,x):
		print("Predicting...")
		if (len(x.shape) == 3) and not(x.shape[0] == self.img_size and x.shape[1] == self.img_size):
			num_across = x.shape[0]//self.img_size
			num_down = x.shape[1]//self.img_size
			tmp_image = np.zeros([x.shape[0]*self.scale,x.shape[1]*self.scale,3])
			for i in range(num_across):
				for j in range(num_down):
					tmp = self.sess.run(self.out,feed_dict={self.input:[x[i*self.img_size:(i+1)*self.img_size,j*self.img_size:(j+1)*self.img_size]]})[0]
					tmp_image[i*tmp.shape[0]:(i+1)*tmp.shape[0],j*tmp.shape[1]:(j+1)*tmp.shape[1]] = tmp
			#this added section fixes bottom right corner when testing
			if (x.shape[0]%self.img_size != 0 and  x.shape[1]%self.img_size != 0):
				tmp = self.sess.run(self.out,feed_dict={self.input:[x[-1*self.img_size:,-1*self.img_size:]]})[0]
				tmp_image[-1*tmp.shape[0]:,-1*tmp.shape[1]:] = tmp
					
			if x.shape[0]%self.img_size != 0:
				for j in range(num_down):
					tmp = self.sess.run(self.out,feed_dict={self.input:[x[-1*self.img_size:,j*self.img_size:(j+1)*self.img_size]]})[0]
					tmp_image[-1*tmp.shape[0]:,j*tmp.shape[1]:(j+1)*tmp.shape[1]] = tmp
			if x.shape[1]%self.img_size != 0:
				for j in range(num_across):
                                        tmp = self.sess.run(self.out,feed_dict={self.input:[x[j*self.img_size:(j+1)*self.img_size,-1*self.img_size:]]})[0]
                                        tmp_image[j*tmp.shape[0]:(j+1)*tmp.shape[0],-1*tmp.shape[1]:] = tmp
			return tmp_image
		else:
			return self.sess.run(self.out,feed_dict={self.input:x})

	"""
	Function to setup your input data pipeline
	"""
	def set_data_fn(self,fn,args,test_set_fn=None,test_set_args=None):
		self.data = fn
		self.args = args
		self.test_data = test_set_fn
		self.test_args = test_set_args

	"""
	Train the neural network
	"""
	def train(self,iterations=1000,save_dir="saved_models"):
		#Removing previous save directory if there is one
		if os.path.exists(save_dir):
			shutil.rmtree(save_dir)
		#Make new save directory
		os.mkdir(save_dir)
		#Just a tf thing, to merge all summaries into one
		merged = tf.summary.merge_all()
		#Using adam optimizer as mentioned in the paper
		optimizer = tf.train.AdamOptimizer()
		#This is the train operation for our objective
		train_op = optimizer.minimize(self.loss)	
		#Operation to initialize all variables
		init = tf.global_variables_initializer()
		print("Begin training...")
		with self.sess as sess:
			#Initialize all variables
			sess.run(init)
			test_exists = self.test_data
			#create summary writer for train
			train_writer = tf.summary.FileWriter(save_dir+"/train",sess.graph)

			#If we're using a test set, include another summary writer for that
			if test_exists:
				test_writer = tf.summary.FileWriter(save_dir+"/test",sess.graph)
				test_x,test_y = self.test_data(*self.test_args)
				test_feed = {self.input:test_x,self.target:test_y}

			#This is our training loop
			for i in tqdm(range(iterations)):
				#Use the data function we were passed to get a batch every iteration
				x,y = self.data(*self.args)
				#Create feed dictionary for the batch
				feed = {
					self.input:x,
					self.target:y
				}
				#Run the train op and calculate the train summary
				summary,_ = sess.run([merged,train_op],feed)
				#If we're testing, don't train on test set. But do calculate summary
				if test_exists:
					t_summary = sess.run(merged,test_feed)
					#Write test summary
					test_writer.add_summary(t_summary,i)
				#Write train summary for this step
				train_writer.add_summary(summary,i)
			#Save our trained model		
			self.save()		
