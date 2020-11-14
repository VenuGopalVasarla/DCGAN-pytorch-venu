# DCGAN-pytorch-venu
My first implementation of DCGAN architecture

## Results on LSUN dataset:
### Results obtained from LSUN 'Bridge' data.
Generated:

![Alt text](https://github.com/VenuGopalVasarla/DCGAN-pytorch-venu/blob/main/data/results/fake_epoch_001.png?raw=true)

Original:

![Alt text](https://github.com/VenuGopalVasarla/DCGAN-pytorch-venu/blob/main/data/results/real_samples.png?raw=true)


## Adding your own dataset:
Place your data in your data/ folder.
Register you dataset in the dataset.py.
add dataset specifications in the config file.

## Rewriting the neural networks:
Add your new neural network in the Networks/ folder.
Use appropriate name for your custom network file.
Synchronize those changes in the config file.

## Adding your training specs:
Using config file you can use define the batch size, number of epochs ... etc.
latent vector size can also be defined.

## Using a pre-trained checkpoint:
Use weights_loader.py to load your weights or checkpoints.
