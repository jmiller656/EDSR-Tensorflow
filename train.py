import data
import argparse
from model import EDSR
parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--imgsize")
parser.add_argument("--scale")
parser.add_argument("--layers")
parser.add_argument("--featuresize")
parser.add_argument("--batchsize")
args = parser.parse_args()
if args.dataset:
	dataset = args.dataset
else:
	dataset = "data/General-100"
data.load_dataset(dataset)
img_size = args.imgsize if args.imgsize else 64
scale = args.scale if args.scale else 2
down_size = img_size/scale
layers = args.layers if args.layers else 10
feature_size = args.featuresize if args.featuresize else 32
batch_size = args.batchsize if args.batchsize else 10
network = EDSR(down_size,layers,feature_size)
network.set_data_fn(data.get_batch,(batch_size,img_size,down_size),data.get_test_set,(img_size,down_size))
network.train()
