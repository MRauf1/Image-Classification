import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop
from os import listdir
from os.path import isfile, join
from PIL import Image


class ImageClassificationDataset(Dataset):
    """ Image Classification dataset """

    def __init__(self, images_folder_path, labels_path, image_size = 227):
        """
        Args:
            images_folder_path (string): Path to the folder with the images.
            labels_path (string): Path to the labels in .txt format.
            image_size (int): Size of the square image, which will be passed to CNN.
        """

        self._images_folder_path = images_folder_path
        self._labels_path = labels_path

        # Normalization values are from ImageNet dataset
        self._image_transforms = Compose([
                Resize((256, 256)),
                CenterCrop((image_size, image_size)),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self._image_names = [file for file in listdir(self._images_folder_path)
                            if isfile(join(self._images_folder_path, file))]
        with open(self._labels_path, "r") as txt_file:
            self._labels = [line.strip() for line in txt_file]


    def __len__(self):
        return len(self._image_names)


    def __getitem__(self, index):

        # Read in the image and resize to 227x227
        image = Image.open(join(self._images_folder_path, self._image_names[index])).convert("RGB")
        image = self._image_transforms(image)

        # For the dataset, the image number is located between indices 15 and 23
        image_number = int(self._image_names[index][15:23])
        label = self._labels[image_number]

        return image, label
