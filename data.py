import scipy.misc
import random
import numpy as np
import os

train_set = []
test_set = []
batch_index = 0

"""
Load set of images in a directory.
This will automatically allocate a 
random 20% of the images as a test set

data_dir: path to directory containing images
"""
def load_dataset(data_dir, img_size):
	"""img_files = os.listdir(data_dir)
	test_size = int(len(img_files)*0.2)
	test_indices = random.sample(range(len(img_files)),test_size)
	for i in range(len(img_files)):
		#img = scipy.misc.imread(data_dir+img_files[i])
		if i in test_indices:
			test_set.append(data_dir+"/"+img_files[i])
		else:
			train_set.append(data_dir+"/"+img_files[i])
	return"""
	global train_set
	global test_set
	imgs = []
	img_files = os.listdir(data_dir)
	for img in img_files:
		try:
			tmp= scipy.misc.imread(data_dir+"/"+img)
			x,y,z = tmp.shape
			coords_x = x / img_size
			coords_y = y/img_size
			coords = [ (q,r) for q in range(coords_x) for r in range(coords_y) ]
			for coord in coords:
				imgs.append((data_dir+"/"+img,coord))
		except:
			print "oops"
	test_size = min(10,int( len(imgs)*0.2))
	random.shuffle(imgs)
	test_set = imgs[:test_size]
	train_set = imgs[test_size:][:200]
	return

"""
Get test set from the loaded dataset

size (optional): if this argument is chosen,
each element of the test set will be cropped
to the first (size x size) pixels in the image.

returns the test set of your data
"""
def get_test_set(original_size,shrunk_size):
	"""for i in range(len(test_set)):
		img = scipy.misc.imread(test_set[i])
		if img.shape:
			img = crop_center(img,original_size,original_size)		
			x_img = scipy.misc.imresize(img,(shrunk_size,shrunk_size))
			y_imgs.append(img)
			x_imgs.append(x_img)"""
	imgs = test_set
	get_image(imgs[0],original_size)
	x = [scipy.misc.imresize(get_image(q,original_size),(shrunk_size,shrunk_size)) for q in imgs]#scipy.misc.imread(q[0])[q[1][0]*original_size:(q[1][0]+1)*original_size,q[1][1]*original_size:(q[1][1]+1)*original_size].resize(shrunk_size,shrunk_size) for q in imgs]
	y = [get_image(q,original_size) for q in imgs]#scipy.misc.imread(q[0])[q[1][0]*original_size:(q[1][0]+1)*original_size,q[1][1]*original_size:(q[1][1]+1)*original_size] for q in imgs]
	return x,y

def get_image(imgtuple,size):
	img = scipy.misc.imread(imgtuple[0])
	x,y = imgtuple[1]
	img = img[x*size:(x+1)*size,y*size:(y+1)*size]
	return img
	

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
	global batch_index
	"""img_indices = random.sample(range(len(train_set)),batch_size)
	for i in range(len(img_indices)):
		index = img_indices[i]
		img = scipy.misc.imread(train_set[index])
		if img.shape:
			img = crop_center(img,original_size,original_size)
			x_img = scipy.misc.imresize(img,(shrunk_size,shrunk_size))
			x.append(x_img)
			y.append(img)"""
	max_counter = len(train_set)/batch_size
	counter = batch_index % max_counter
	window = [x for x in range(counter*batch_size,(counter+1)*batch_size)]
	imgs = [train_set[q] for q in window]
	x = [scipy.misc.imresize(get_image(q,original_size),(shrunk_size,shrunk_size)) for q in imgs]#scipy.misc.imread(q[0])[q[1][0]*original_size:(q[1][0]+1)*original_size,q[1][1]*original_size:(q[1][1]+1)*original_size].resize(shrunk_size,shrunk_size) for q in imgs]
	y = [get_image(q,original_size) for q in imgs]#scipy.misc.imread(q[0])[q[1][0]*original_size:(q[1][0]+1)*original_size,q[1][1]*original_size:(q[1][1]+1)*original_size] for q in imgs]
	batch_index = (batch_index+1)%max_counter
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
	startx = random.sample(range(x-cropx-1),1)[0]#x//2-(cropx//2)
	starty = random.sample(range(y-cropy-1),1)[0]#y//2-(cropy//2)
	return img[starty:starty+cropy,startx:startx+cropx]





