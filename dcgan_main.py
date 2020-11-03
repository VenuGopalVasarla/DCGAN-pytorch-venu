"""
    Created by Venu Gopal Vasarla.
    This is the main file from where dcgan is called.
    View README in order to use the repo and run DCGAN.
"""

from __future__ import print_function
from datasets import Dataset
from config import device
from networks.simple_networks import Generator, Discriminator
from weights_loader import weights_init

# Dataset loading
data_loader = Dataset()
loaded_data = data_loader.get_data()

# Calling networks
generator = Generator().to(device)
generator.apply(weights_init)
print(type(generator))
