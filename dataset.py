import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop
from os import listdir
from os.path import isfile, join
from PIL import Image


class ImageClassificationDataset(Dataset):
    """ Image Classification dataset """

    def __init__(self, images_folder_path, labels_path, mode, image_size = 227,
                        train_val_test_ratio = [0.7, 0.1, 0.2]):
        """
        Args:
            images_folder_path (string): Path to the folder with the images.
            labels_path (string): Path to the labels in .txt format.
            image_size (int): Size of the square image, which will be passed to CNN.
        """

        assert(sum(train_val_test_ratio) == 1.0)
        assert(mode == "train" or mode == "val" or mode == "test")

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

        num_train = int(len(self._image_names) * train_val_test_ratio[0])
        num_val = int(len(self._image_names) * train_val_test_ratio[1])
        num_test = int(len(self._image_names) * train_val_test_ratio[2])

        if(mode == "train"):
            self._image_names = self._image_names[:num_train]
            self._labels = self._labels[:num_train]
        elif(mode == "val"):
            self._image_names = self._image_names[num_train:(num_train + num_val)]
            self._labels = self._labels[num_train:(num_train + num_val)]
        elif(mode == "test"):
            self._image_names = self._image_names[(num_train + num_val):]
            self._labels = self._labels[(num_train + num_val):]


    def __len__(self):
        return len(self._image_names)


    def __getitem__(self, index):

        label_num = self._labels[index]
        label = torch.tensor([int(label_num)], dtype = torch.int64)

        image_name = "ILSVRC2012_val_" + str(label_num.zfill(8)) + ".JPEG"
        image = Image.open(join(self._images_folder_path, image_name)).convert("RGB")
        image = self._image_transforms(image)

        return image, label
