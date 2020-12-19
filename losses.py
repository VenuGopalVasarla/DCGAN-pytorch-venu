"""
    Created by Venu Gopal Vasarla.
    Script to use different loss functions, which can be integrated to advanced
    trainer.
    Also implemented soft labels and noisy labels.
"""
import torch
from advanced_config import use_soft_labels, real_range, fake_range


def load_labels(shape, is_real=True, use_noise=False):
    """
        A function to load the labels for loss functions.
        Performs softening, adding noise to the labels also.
        Args:
            shape: shape of the input tensor.
            is_real: whether to load a real value label or not. default: true
            use_noise: flips the label values to fool discriminator. default: False
        Returns:
            a labels tensor of size 'shape' with required characteristics.
    """
    if use_noise:
        is_real = not is_real
    if use_soft_labels:
        if is_real:
            label = torch.FloatTensor(shape).uniform_(real_range[0], real_range[1])
        else:
            label = torch.FloatTensor(shape).uniform_(fake_range[0], fake_range[1])
    else:
        if is_real:
            label = torch.full(shape, 1)
        else:
            label = torch.full(shape, 0)

    return label


def minmax_dis_loss(real_output=None, fake_output=None, type='mean', use_noise=False):
    """
        caluclates the minmax loss for the discriminator using the equation:
            losses =  L[l1,l2,l3,...ln]
            ln = log[D(Xn)] + log[1 - D(Xn')]
            if type is none then output is losses
            if type is 'mean' then output is avg(losses)
            if type is 'sum' then output is sum(losses)
        Args:
            real_output: output of discriminator when input is real one.
            fake_output: output of discriminator when input is fake one.
            type: the output style for the losses. ['mean', 'sum', 'none']
        Output:
            a list of losses values/ mean of all the losses/ sum of all losses.
    """

    data_present = False
    real_dis_loss = 0
    if real_output is not None:
        labels = load_labels(real_output.size(), is_real=True, use_noise=use_noise)

        # cleaning values to replace 0. with 0.000001
        # real_output = torch.where(real_output != 0., real_output, 0.000001)

        # implementing Loss formula
        real_dis_loss = max(labels * torch.log(real_output), -100)
        data_present = True

    fake_dis_loss = 0
    if fake_output is not None:
        labels = load_labels(fake_output.size(), is_real=False, use_noise=use_noise)

        # cleaning values to replace 1 with 0.999999
        # fake_output = torch.where(fake_output != 1, fake_output, 0.999999)

        # implementing Loss formula
        fake_dis_loss = max(labels * torch.log(torch.full(fake_output.shape, 1.) - fake_output), -100)
        data_present = True

    if data_present:
        final_loss = real_dis_loss + fake_dis_loss
        if type == 'mean':
            return final_loss.mean()
        elif type == 'sum':
            return final_loss.sum()
        return final_loss
    raise Exception("No input/None provided to the loss function")
