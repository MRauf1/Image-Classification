import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join


class ImageClassificationDataset(Dataset):
    """ Image Classification dataset """

    def __init__(self, images_folder_path, labels_path):
        """
        Args:
            images_folder_path (string): Path to the folder with the images.
            labels_path (string): Path to the labels in .txt format.
        """

        self._images_folder_path = images_folder_path
        self._labels_path = labels_path
        self._image_names = [file for file in listdir(self._images_folder_path)
                            if isfile(join(self._images_folder_path, file))]
        with open(self._labels_path, "r") as txt_file:
            self._labels = [line.strip() for line in txt_file]


    def __len__(self):
        return len(self._image_names)

    def __getitem__(self, index):
        pass
