"""
    This file is created by Venu Gopal Vasarla.

    Datasets are integrated in this file.
    You can use your own dataset here creating your own class.
"""
from torchvision import dataset as dset
from torchvision import transforms as transforms


class Dataset:
    """
        This Dataset module returns dataset object from torchvision.
        args :
        dataset_name: The name of the dataset used ['LSUN']
        dataset_classes: list of classes [None,{'train', 'val', 'test'},[]]
    """

    def __init__(self, dataset_name='LSUN',
                 dataset_classes=None,
                 ):
        self.dataset_name = dataset_name
        self.dataset_classes = dataset_classes

    def from_dataset(self):
        if self.dataset_name == 'LSUN':
            self.lsun(self.dataset_classes)

    def lsun(self):
        if type(self.dataset_classes) is list:
            updated_classes = [c + '_train' for c in self.dataset_classes]
        elif isinstance(self.dataset_classes, str):
            updated_classes = self.dataset_classes
        else:
            updated_classes = 'train'
        dataset = dset.LSUN(root=opt.dataroot, classes=updated_classes,
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5)),
                            ]))
