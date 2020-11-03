"""
    Created by Venu Gopal Vasarla.
    This file contains all the paths, external values and others,
    to modify datasets, models, networks and all.
"""
import os

# data configurations:
dataset = 'LSUN'  # dataset name to be provided here; default: 'LSUN'
dataset_path = 'data/lsun/'  # provide the input data path; default:'data/lsun'
dataset_classes = ['bridge']  # classes for dataset; default: '[bridge]'
image_size = 64  # image size for the N/Ws; default: 64
center_crop = 64  # center crops for dataset transform; default: 64
out_dir = 'data/results'  # output path to save results; default: 'data/out'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(dataset_path):
    raise Exception('The dataset path does not exist. Provide a valid path')

# train configurations:
workers = 2  # num of workers used for loading data; default: 2
batch_size = 64  # input batch size to use; default: 64
latent_vector = 100  # size of input latent vector; default: 100
gen_size = 64  # size of outputs for generator to be considered; default: 64
dis_size = 64  # size ofinputs for discriminator; default: 64
CPU = False  # If training is to be done on CPU.
num_gpus = 1  # num of GPUs to use; default: 1

if CPU:
    device = "cpu"
else:
    device = "cuda:0"
