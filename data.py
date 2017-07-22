import scipy.misc
import random
import os

train_set = []
test_set = []

"""
Load set of images in a directory.
This will automatically allocate a 
random 20% of the images as a test set

data_dir: path to directory containing images
"""
def load_dataset(data_dir):
	img_files = os.listdir(data_dir)
	test_size = int(len(img_files)*0.2)
	test_indices = random.sample(range(len(img_files)),test_size)
	for i in range(len(img_files)):
		#img = scipy.misc.imread(data_dir+img_files[i])
		if i in test_indices:
			test_set.append(data_dir+"/"+img_files[i])
		else:
			train_set.append(data_dir+"/"+img_files[i])
	return

"""
Get test set from the loaded dataset

size (optional): if this argument is chosen,
each element of the test set will be cropped
to the first (size x size) pixels in the image.

returns the test set of your data
"""
def get_test_set(original_size,shrunk_size):
	y_imgs = []
	x_imgs = []
	for i in range(len(test_set)):
		img = scipy.misc.imread(test_set[i])
		img = crop_center(img,original_size,original_size)
		x_img = scipy.misc.imresize(img,(shrunk_size,shrunk_size))
		y_imgs.append(img)
		x_imgs.append(x_img)
	return x_imgs,y_imgs

"""
Get a batch of images from the training
set of images.

batch_size: size of the batch
original_size: size for target images
shrunk_size: size for shrunk images

returns x,y where:
	-x is the input set of shape [-1,shrunk_size,shrunk_size,channels]
	-y is the target set of shape [-1,original_size,original_size,channels]
"""
def get_batch(batch_size,original_size,shrunk_size):
	x =[]
	y =[]
	img_indices = random.sample(range(len(train_set)),batch_size)
	for i in range(len(img_indices)):
		index = img_indices[i]
		img = scipy.misc.imread(train_set[index])
		img = crop_center(img,original_size,original_size)
		x_img = scipy.misc.imresize(img,(shrunk_size,shrunk_size))
		x.append(x_img)
		y.append(img)
	return x,y

"""
Simple method to crop center of image

img: image to crop
cropx: width of crop
cropy: height of crop
returns cropped image
"""
def crop_center(img,cropx,cropy):
	y,x,_ = img.shape
	startx = x//2-(cropx//2)
	starty = y//2-(cropy//2)    
	return img[starty:starty+cropy,startx:startx+cropx]




