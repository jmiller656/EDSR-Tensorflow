import data
from model import EDSR
data.load_dataset("data/General-100")
img_size = 64
down_size = 32
layers = 10
feature_size = 32
batch_size = 10
network = EDSR(down_size,layers,feature_size)
network.set_data_fn(data.get_batch,(batch_size,img_size,down_size),data.get_test_set,(img_size,down_size))
network.train()
