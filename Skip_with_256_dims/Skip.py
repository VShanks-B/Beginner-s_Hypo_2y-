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
dimInt = (number_of_points-kernel_size1[0]+1-kernel_size2[0]+1)


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        
        self.lin1 = nn.LazyLinear(4096)
        self.lin2 = nn.Linear(4096, 2048, bias=False)
        self.lin3 = nn.Linear(2048, 4096, bias=False)  
        
        # Skip connection
        self.skip_connection = nn.Identity()  # Identity function initially, will be updated in forward method
        
        self.fc1_mean = nn.Linear(4096, latent_dim)
        self.fc1_logvar = nn.Linear(4096, latent_dim)

    def forward(self, x):                
        x = x.view(x.size(0), -1)  # Flatten the feature map
        
        x = F.relu(self.lin1(x))
        
        x_skip = self.skip_connection(x)  # Apply skip connection
        
        x = F.relu(self.lin2(x))  # Now matches input size of lin3
        
        x = F.relu(self.lin3(x) + x_skip)  # Apply skip connection
        
        z_mean = self.fc1_mean(x)
        z_logvar = self.fc1_logvar(x)
        
        return z_mean, z_logvar


class Sampling(nn.Module):
    def __init__(self, epsilon_std=1e-6):
        super(Sampling, self).__init__()
        self.epsilon_std = epsilon_std

    def forward(self, z_mean, z_logvar):
        batch_size, latent_dim = z_mean.size()
        epsilon = torch.randn_like(z_mean) * self.epsilon_std  # Add small noise for numerical stability
        std = torch.exp(0.5 * z_logvar) + self.epsilon_std  # Add epsilon for numerical stability
        z = z_mean + std * epsilon
        return z.detach()  # Detach to prevent gradient flow


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        with open("config_model.yaml", "r") as file:
            config = yaml.safe_load(file)
        number_of_functions = config["number_of_functions"]
        self.fc1 = nn.LazyLinear(2048)
        self.fc2 = nn.LazyLinear(out_channels2 * dimInt)
        self.conv_transpose1 = nn.ConvTranspose2d(out_channels2, 32, kernel_size=kernel_size1, bias=bias)
        self.conv1 = nn.Conv2d(32, 64, kernel_size1, bias=bias)
        self.conv_transpose2 = nn.ConvTranspose2d(64, out_channels1, kernel_size=kernel_size1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels1, 64, kernel_size2, bias=bias)
        self.conv_transpose3 = nn.ConvTranspose2d(64, in_channels1, kernel_size=kernel_size2, bias=bias)
        self.conv_transpose4 = nn.ConvTranspose2d(in_channels1, in_channels1, kernel_size=kernel_size2, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        batch_size, latent_dim = z.size()
        
        x = torch.reshape(x, (batch_size, out_channels2, dimInt, 1))  # Reshape to (batch_size, channels, height, width)
                
        skip_con = x
        
        x = self.conv_transpose1(x)
        
        x = F.relu(self.conv1(x)+skip_con)
                
        x = self.conv_transpose2(x)
        
        x = F.relu(self.conv2(x))
        
        x = self.conv_transpose3(x)
        
        x = self.conv_transpose4(x)

        return x


# define the CVAE model
class CVAE(nn.Module):
    def __init__(self,latent_dim = 8):
        super(CVAE, self).__init__()
        
        self.encoder = Encoder(latent_dim)
        self.sampling = Sampling()
        self.decoder = Decoder(latent_dim)

    def forward(self, x, cond):
        z_mean, z_logvar = self.encoder(x)
        z = self.sampling(z_mean, z_logvar)
        x_recon = self.decoder(z, cond)
        return x_recon, z_mean, z_logvar
