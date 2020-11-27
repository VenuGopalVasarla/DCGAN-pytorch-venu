import torch
import os
import re


def weights_init(m):
    """ Weights initalizer"""

    # TODO: add script to read a checkpoint.
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def from_previous_ckpt(network, checkpoint):
    """
        Loads the pretrained weights into model.
        Used to continue training.
        Args:
            network: Network to apply weights.
            checkpoint: path to the checkpoint.
        Returns:
            network with applied weights.
    """
    if os.path.exists(checkpoint):
        if os.path.isfile(checkpoint):
            try:
                network.load_state_dict(torch.load(checkpoint))
                print(f"Loaded weights from {checkpoint}")
            except Exception as e:
                print(e)
                print(f"{checkpoint} is a invalid checkpoint")
        if os.path.isdir(checkpoint):
            epoch = 0
            for ckpt in os.listdir(checkpoint):
                if ckpt[-3:] == '.pth':
                    try:
                        tmp_int_list = re.findall('[0-9]+', ckpt)
                        ckpt_epoch = int(tmp_int_list[-1])
                    except IndexError:
                        ckpt_epoch = 0
                    if ckpt_epoch >= epoch:
                        epoch = ckpt_epoch
                        file_name = os.join(checkpoint, ckpt)
            if file_name:
                try:
                    network.load_state_dict(torch.load(file_name))
                    print(f"Loaded weights from {file_name}")
                except Exception as e:
                    print(e)
                    print(f"{file_name} is a invalid checkpoint")
            else:
                print(f"No checkpoint found in {checkpoint}")

    else:
        print(f"the checkpoint path: {checkpoint} doesn't exist.")
        print("Neglecting this checkpoint.")
