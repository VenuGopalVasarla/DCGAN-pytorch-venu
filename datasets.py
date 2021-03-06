"""
    This file is created by Venu Gopal Vasarla.

    All the datasets for DCGAN are to be integrated in this file.
    You can use your own dataset here creating your own Dataset method.

    Please follow the instructions in order to use your own dataset:
    1. Make sure to add an new argument type in method 'from_dataset' with your
        dataset name.
    2. Now create a new method in Dataset calling torchvision's dataset on your
        dataset folder.
    3. Apply required transformations, and you can also add an argument for
        your required classes.
    4. Please donot forget to include the changes in the config file.
"""
import torch
from torchvision import transforms as transforms
from torchvision import datasets as dset
from config import dataset, dataset_path, dataset_classes, image_size, \
    center_crop, batch_size, workers, download_mnist


class Dataset:
    """
        This Dataset module returns dataset object from torchvision.
        args :
        dataset_name: The name of the dataset used ['LSUN']
        dataset_classes: list of classes [None,{'train', 'val', 'test'},[]]
    """

    def __init__(self):
        if dataset:
            self.dataset_name = dataset
        else:
            self.dataset_name = 'LSUN'
        assert dataset_path
        self.data_path = dataset_path
        if dataset_classes:
            self.dataset_classes = dataset_classes
        else:
            self.dataset_classes = None
        if image_size:
            self.image_size = image_size
        else:
            self.image_size = 64
        if center_crop:
            self.center_crop = center_crop
        else:
            self.center_crop = 64
        self.download = download_mnist

    def get_data(self):
        """
            This method calls the specified method for a certain dataset.
            Please don't forget to call your dataset method here.
        """
        if self.dataset_name == 'LSUN':
            dataset = self.lsun()

        elif self.dataset_name == 'lfw':
            dataset = self.lfw()

        elif self.dataset_name == 'mnist':
            dataset = self.mnist()

        elif self.dataset_name == 'cifar10':
            dataset = self.cifar10()

        else:
            print(f'Invalid dataset name: {self.dataset_name}')
            raise Exception('Dataset error: Choose a valid dataset.')

        return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=int(workers))

    def lsun(self):
        """
            This method returns a set of 'LSUN' dataset.
            Can produce required classes or sets in 'LSUN'.
            Please go through the 'LSUN' dataset website for more info.
        """
        if type(self.dataset_classes) is list:
            updated_classes = [c + '_train' for c in self.dataset_classes]
        elif isinstance(self.dataset_classes, str):
            updated_classes = self.dataset_classes
        else:
            updated_classes = 'train'
        dataset = dset.LSUN(root=self.data_path, classes=updated_classes,
                            transform=transforms.Compose([
                                transforms.Resize(self.image_size),
                                transforms.CenterCrop(self.center_crop),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5)),
                            ]))
        return dataset

    def lfw(self):
        """
            this method returns a set of 'lsw' dataset.
            Please go through lfw dataset website for more info.
        """
        dataset = dset.ImageFolder(root=self.data_path,
                                   transform=transforms.Compose([
                                       transforms.Resize(self.image_size),
                                       transforms.CenterCrop(self.center_crop),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5)),
                                   ]))
        return dataset

    def mnist(self):
        """
            This method returns a set of 'MNIST' dataset.
            Please go throuh mnist website for more info.
        """
        dataset = dset.MNIST(root=self.data_path, download=self.download,
                             transform=transforms.Compose([
                                 transforms.Resize(image_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,)),
                             ]))
        return dataset

    def cifar10(self):
        """
            This method returns a set of 'CIFAR10' dataset.
            Please go throuh cifar10 website for more info.
        """
        dataset = dset.CIFAR10(root=self.data_path, download=self.download,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5)),
                               ]))
        return dataset
