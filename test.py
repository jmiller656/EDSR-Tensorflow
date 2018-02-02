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
down_size = args.imgsize//args.scale
network = EDSR(down_size,args.layers,args.featuresize,scale=args.scale)
network.resume(args.savedir)
if args.image:
	x = scipy.misc.imread(args.image)
else:
	print("No image argument given")
inputs = x
outputs = network.predict(x)
if args.image:
	scipy.misc.imsave(args.outdir+"/input_"+args.image,inputs)
	scipy.misc.imsave(args.outdir+"/output_"+args.image,outputs)
