import torch
from torch.utils.data import Dataset


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


    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
