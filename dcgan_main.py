"""
    Created by Venu Gopal Vasarla.
    This is the main file from where dcgan is called.
    View README in order to use the repo and run DCGAN.
"""

from __future__ import print_function
from datasets import Dataset
from config import device, summarize_nets, num_epochs, gen_ckpt_path, \
    dis_ckpt_path
from networks.simple_networks import Generator, Discriminator
from weights_loader import weights_init, from_previous_ckpt
from trainer import Trainer

# loading dataset
data_loader = Dataset()
loaded_data = data_loader.get_data()

# Calling networks
generator = Generator().to(device)
generator.apply(weights_init)
from_previous_ckpt(generator, gen_ckpt_path)

discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
from_previous_ckpt(discriminator, dis_ckpt_path)

if summarize_nets:
    print("\t\t\t\t\t******Generator******\n", generator)
    print("\n\t\t\t\t\t******Discriminator******\n", discriminator)

# Training:
gan_train = Trainer(gen=generator, dis=discriminator, device=device)

gan_train.train(num_iters=num_epochs, data=loaded_data)
