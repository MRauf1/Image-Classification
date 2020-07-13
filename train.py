from dataset import ImageClassificationDataset
from utils import read_config
from models.alexnet import AlexNet
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm.auto import tqdm


def train(train_dataset, val_dataset, configs):

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = configs["batch_size"],
            shuffle = True
    )

    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = configs["batch_size"],
            shuffle = False
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = model.parameters(), lr = configs["lr"])

    for epoch in range(configs["epochs"]):

        model.train()
        running_loss = 0.0
        correct = 0

        for i, (inputs, labels) in tqdm(enumerate(train_loader)):

            inputs, labels = inputs.to(device), labels.squeeze().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

        print("[%d] loss: %.4f" %
                  (epoch + 1, running_loss / train_dataset.__len__()))

        model.eval()
        correct = 0

        with torch.no_grad():

            for i, (inputs, labels) in tqdm(enumerate(val_loader)):

                inputs, labels = inputs.to(device), labels.squeeze().to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        print("Accuracy of the network on the %d test images: %.4f %%" %
                (val_dataset.__len__(), 100. * correct / val_dataset.__len__()))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Image Classification")
    parser.add_argument("--config_path", default = "config.yaml", help = "Path to YAML config file.")
    args = parser.parse_args()

    configs = read_config(args.config_path)

    train_dataset = ImageClassificationDataset(
                configs["images_folder_path"],
                configs["labels_path"],
                "train"
    )

    val_dataset = ImageClassificationDataset(
                configs["images_folder_path"],
                configs["labels_path"],
                "val"
    )

    train(train_dataset, val_dataset, configs)
