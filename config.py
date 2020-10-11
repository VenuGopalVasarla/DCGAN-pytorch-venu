"""
    Created by Venu Gopal Vasarla.
    This file contains all the paths, external values and others,
    to modify datasets, models, networks and all.
"""
import os

# data configurations
dataset = 'LSUN'
dataset_path = 'data/lsun/'
dataset_classes = False
image_size = 64
center_crop = 64
out_dir = 'data/out'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# train configurations
workers = 2
batch_size = 64
