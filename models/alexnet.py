import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    
    def __init__(self):

        super(AlexNet, self).__init__()

        self._conv = nn.Sequential(

            nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride = 4),
            nn.ReLU(),
            nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75, k = 2),
            nn.MaxPool2d(kernel_size = 3, stride = 2),

            nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75, k = 2),
            nn.MaxPool2d(kernel_size = 3, stride = 2),

            nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, padding = 1),
            nn.ReLU(),

            nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, padding = 1),
            nn.ReLU(),

            nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)

        )

        self._fc = nn.Sequential(

            nn.Dropout(p = 0.5),
            nn.Linear(in_features = 4096, out_features = 4096),
            nn.ReLU(),

            nn.Dropout(p = 0.5),
            nn.Linear(in_features = 4096, out_features = 4096),
            nn.ReLU(),

            nn.Linear(in_features = 4096, out_features = 1000)

        )

        self.init_parameters()


    def init_parameters(self):
        """ Initialize the weights and biases as described in the paper """
        for layer in self._conv:
            if(isinstance(layer, nn.Conv2d)):
                nn.init.normal_(layer.weight, mean = 0, std = 0.01)
                nn.init.constant_(layer.bias, 0)

        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

        for layer in self._fc:
            if(isinstance(layer, nn.Linear)):
                nn.init.constant_(layer.bias, 1)


    def forward(self, x):
        x = self._conv(x)
        x = x.view(-1, 4096)
        x = self._fc(x)
        return x
