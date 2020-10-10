# MNIST Classifier in PyTorch
# Tutorial kinda followed by b0kch01

import torch # Pytorch module
import torchvision # Where to get dataset
import torchvision.transforms as tfs # Normalize dataset for good results

from PIL import Image
import matplotlib.pyplot as plt

# Dataset Processing

# Compose will chain multiple transforms together
transformer = tfs.Compose([
	tfs.ToTensor(), # Turn image into a tensor (use GPU)
	tfs.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)) # (Normalize to [-1, 1])
])

# Downloading Raw datset
train_data = torchvision.datasets.MNIST(
	root = "./dataset", # Where the dataset will get downloaded
	train = True, # True bc this is the training set
	download = True, # Yes, download the dataset
	transform = None # No transformations needed yet
)

test_data = torchvision.datasets.MNIST(
	root = "./dataset", # Where the dataset will get downloaded
	train = False, # False bc this is the testing set
	download = True, # Yes, download the dataset
	transform = None # No transformations needed yet
)

# Create a dataloader (a dataset class by pytorch)
from torch.utils.data import DataLoader

trainloader = DataLoader(
	train_data, # Takes in the dataset
	batch_size = 1, # Batch number
	shuffle = True, # Randomize images after each epoch
	num_workers=4 # How many different processes run at once 
) 

testloader = DataLoader(
	test_data, # Takes in the dataset
	batch_size = 1, # Batch number
	shuffle = True, # Randomize images after each epoch
	num_workers=4 # How many different processes run at once 
)

# CONSTRUCTION!
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		