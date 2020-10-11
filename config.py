"""
    Created by Venu Gopal Vasarla.
    This file contains all the paths, external values and others,
    to modify datasets, models, networks and all.
"""
import os

# data configurations
dataset = 'LSUN'  # Dataset name to be provided here; default: 'LSUN'
dataset_path = 'data/lsun/'  # Provide the input data path; default:'data/lsun'
dataset_classes = ['bridge']  # classes for dataset; default: '[bridge]'
image_size = 64  # image size for the N/Ws; default: 64
center_crop = 64  # Center crops for dataset transform; default: 64
out_dir = 'data/out'  # Output path to save results; default: 'data/out'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(dataset_path):
    raise Exception('This path does not exist. Please provide a valid path')

# train configurations
workers = 2
batch_size = 64
