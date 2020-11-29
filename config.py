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
num_channels = 3  # number of channels for each image, for mnist: 1
download_mnist = True  # whether to download mnist data or not.

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(dataset_path):
    raise Exception('The dataset path does not exist. Provide a valid path')
num_classes = len(dataset_classes)

# train configurations:
workers = 2  # num of workers used for loading data; default: 2
batch_size = 64  # input batch size to use; default: 64
latent_vector = 100  # size of input latent vector; default: 100
gen_size = 64  # size of outputs for generator to be considered; default: 64
dis_size = 64  # size ofinputs for discriminator; default: 64
summarize_nets = True  # Give details of models architectures
CPU = False  # If training is to be done using CPU
num_gpus = 1  # num of GPUs to use; default: 1
loss = "BCE"  # loss to be used. default: Binary Cross Entropy
optimizer = "adam"  # optimizer to be used. default: Adam
lr = 0.0002  # learning rate. default: 0.0002
beta = 0.5  # beta1 value for the optimizer, default: 0.5
num_epochs = 10  # Number of iterations to be done
dis_ckpt_path = "data/results/discriminator_ckpts/"  # Path to the pre-trained ckpt if available
gen_ckpt_path = "data/results/generator_ckpts/"  # Path to the pre-trained ckpt if available

if CPU:
    device = "cpu"
else:
    device = "cuda:0"
