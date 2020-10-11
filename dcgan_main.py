"""
    Created by Venu Gopal Vasarla.
    This is the main file from where dcgan is called.
    View README in order to use the repo and run DCGAN.
"""

from __future__ import print_function
from datasets import Dataset

# Dataset loading
data_loader = Dataset()
loaded_data = data_loader.get_data()
