import data
import argparse
from model import EDSR
parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
args = parser.parse_args()
if args.dataset:
	dataset = args.dataset
else:
	dataset = "data/General-100"
data.load_dataset(dataset)
img_size = 64
down_size = 32
layers = 10
feature_size = 32
batch_size = 10
network = EDSR(down_size,layers,feature_size)
network.set_data_fn(data.get_batch,(batch_size,img_size,down_size),data.get_test_set,(img_size,down_size))
network.train()
