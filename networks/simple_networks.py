import torch.nn as nn
from config import latent_vector, gen_size, num_channels, num_gpus, dis_size


class Generator(nn.Module):
    """
        This class creates and a feed forward simple generator neural network.
        Uses ReLU transforms and batchnorm after every conv(+transpose) layer.
        inputs:
            num_gpu: number of gpus used, default: 1
            input: [for forward method] input to the neural net.
        outputs:
            output tensor after feed it the network.
    """

    def __init__(self, num_gpu=num_gpus):
        super(Generator, self).__init__()
        self.num_gpu = num_gpu
        self.gen_net = nn.Sequential(
            nn.ConvTranspose2d(latent_vector, gen_size * 8,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(gen_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_size * 8, gen_size * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(gen_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_size * 4, gen_size * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(gen_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_size * 2, gen_size,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(gen_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_size, num_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        """
            This initializes the nn module to use GPUs allocated,
            and then runs the input through the network.
            inputs: input tensor.
            outputs: output tensor.
        """
        if input.is_cuda and self.num_gpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.num_gpu))
        else:
            output = self.gen_net(input)
        return output


class Discriminator(nn.Module):
    """
        This class creates and a simple discriminator neural network.
        Uses leaky ReLU transforms and batchnorm after every conv layer.
        inputs:
            num_gpu: number of gpus used, default: 1
            input: [for forward method] input to the neural net.
        outputs:
            output tensor after feed it the network.
    """

    def __init__(self, num_gpu=num_gpus):
        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu
        self.dis_net = nn.Sequential(
            nn.Conv2d(num_channels, dis_size,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dis_size, dis_size * 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(dis_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dis_size * 2, dis_size * 4,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(dis_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dis_size * 4, dis_size * 8,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(dis_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dis_size * 8, 1,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
            This initializes the nn module to use GPUs allocated,
            and then runs the input through the network.
            inputs: input tensor.
            outputs: output tensor.
        """
        if input.is_cuda and self.num_gpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.num_gpu))
        else:
            output = self.dis_net(input)

        return output.view(-1, 1).squeeze(1)
