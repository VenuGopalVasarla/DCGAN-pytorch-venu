import torch
import torchvision.utils as vutils
from config import out_dir


def save_checkpoint(state_dictionary, name, epoch, output_dir=out_dir):
    """
        Saves the pytorch checkpoint when called.
        Args:
            state_dictionary: The dict to be saved as checkpoint.
            name: name of the checkpoint path.
            epoch: epoch number of the checkpoint.
            output_dir: the output directory to be saved.
        Output:
            A torch ckpt at the output dir.
    """
    file_name = output_dir + '/' + name + '_' + str(epoch) + '.pth'
    torch.save(state_dictionary, file_name)

    print(f'saved checkpoint of {epoch} epoch to {file_name}')


def save_images(data, name, epoch, output_dir=out_dir):
    """
        Saves the image when called.
        Args:
            data: The torch image data.
            name: name of the image type.
            epoch: epoch number of the image.
            output_dir: the output directory to be saved.
        Output:
            A .png file at the output dir.
    """
    file_name = output_dir + '/' + name + '_' + str(epoch) + '.png'
    vutils.save_image(data, file_name, normalize=True)

    print(f"image at {epoch} epoch saved to {file_name}")
