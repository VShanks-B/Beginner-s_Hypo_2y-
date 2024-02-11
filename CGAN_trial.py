import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader 

import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.insert(0,'/kaggle/input/hypo-stuff')

# Load the config YAML file
import yaml
# with open("config_model.yaml", "r") as file:
with open("/kaggle/input/hypo-stuff/config_model.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract the parameters from the config
number_of_points = config["number_of_points"]
number_of_functions = config["number_of_functions"]

bias = config["bias"]
in_channels1 = config["in_channels1"]
out_channels1 = config["out_channels1"]
kernel_size1 = config["kernel_size1"]
out_channels2 = config["out_channels2"]
kernel_size2 = config["kernel_size2"]
#latent_dim = config["latent_dim"]

# define the input dimensions
input_shape = (1, number_of_points, 2)
output_shape = (1, number_of_points, 2)


#Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        
        self.input_size = input_shape
        self.output_size = output_shape
        
        self.fc = nn.Linear(input_size, 128 ,bias=bias)
        self.fc2 = nn.Linear(128, 256,bias=bias)
        self.fc3 = nn.Linear(256, output_size,bias=bias)
        self.activation = nn.ReLU()
        
    def forward(self, z, condition):
        x = torch.cat([z, condition], dim=1)  # Concatenate noise and condition
        x = self.activation(self.fc(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

#Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size, condition_size):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.condition_size = condition_size
        
        self.fc1 = nn.Linear(input_size + condition_size, 128,bias=bias)
        self.fc2 = nn.Linear(128, 64,bias=bias)
        self.fc3 = nn.Linear(64, 1,bias=bias)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)  # Concatenate input and condition
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)