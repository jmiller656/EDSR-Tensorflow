from model import EDSR
import scipy.misc
import argparse
import data
import os
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/General-100")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=10,type=int)
parser.add_argument("--savedir",default="saved_models")
parser.add_argument("--iterations",default=1000,type=int)
parser.add_argument("--numimgs",default=5,type=int)
parser.add_argument("--outdir",default="out")
parser.add_argument("--image")
args = parser.parse_args()
if not os.path.exists(args.outdir):
	os.mkdir(args.outdir)
data.load_dataset(args.dataset)
down_size = args.imgsize/args.scale
network = EDSR(down_size,args.layers,args.featuresize,scale=args.scale)
network.resume(args.savedir)
if args.image:
	y = data.crop_center(scipy.misc.imread(args.image),args.imgsize,args.imgsize)
	x = [scipy.misc.imresize(y,(down_size,down_size))]
	y = [y]
else:
	x,y=data.get_batch(args.numimgs,args.imgsize,down_size)
inputs = x
outputs = network.predict(x)
correct = y
if args.image:
	scipy.misc.imsave(args.outdir+"/input"+args.image,inputs[0])
	scipy.misc.imsave(args.outdir+"/output"+args.image,outputs[0])
	scipy.misc.imsave(args.outdir+"/correct"+args.image,correct[0])
else:
	for i in range(len(inputs)):
		scipy.misc.imsave(args.outdir+"/input"+str(i)+".png",inputs[i])
	for i in range(len(outputs)):
		scipy.misc.imsave(args.outdir+"/output"+str(i)+".png",outputs[i])
	for i in range(len(correct)):
		scipy.misc.imsave(args.outdir+"/correct"+str(i)+".png",correct[i])
