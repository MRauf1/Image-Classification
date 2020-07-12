from dataset import ImageClassificationDataset
import torch




if __name__ == "__main__":

    dataset = ImageClassificationDataset("ImageNet/ILSVRC2012_img_val", "ImageNet/ILSVRC2012_validation_ground_truth.txt",
                mode = "train")
    print(dataset.__len__())
