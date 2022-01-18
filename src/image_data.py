from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class ImageData(Dataset):
    """
    Custom pytorch dataset.
    """
    def __init__(self, files, labels=None, force_rgb=False, transform=None):
        """

        Parameters
        ----------
        files: list[str] list of files
        labels: list[int] list of labels
        force_rgb: bool, forces gray scale images to have 3 channels.
        transform: pytorch transform
        """
        self.files = files
        self.transform = transform
        self.labels = labels
        if labels is not None:
            self.classes = list(set(labels))
        self.force_rgb = force_rgb

    def __len__(self):
        """

        Returns
        -------
        the length of the dataset
        """
        return len(self.files)

    def __getitem__(self, index):
        """
        Return an image from the dataset

        Parameters
        ----------
        index: int index of the element in the dataset.

        Returns
        -------
        either image or (image,label)
        """
        # Read image
        image = np.asarray(Image.open(self.files[index]))

        # Force image to have 3 channels
        if len(image.shape) == 2 and self.force_rgb:
            image = np.stack((image,) * 3, axis=-1)

        # Make it a PIL image as required by pytorch
        image = Image.fromarray(np.uint8(image))

        # Transform the image
        if self.transform is not None:
            image = self.transform(image)

        # Return image if no label is available
        if self.labels is None:
            return image

        # Return image/label pair.
        return image, self.labels[index]
