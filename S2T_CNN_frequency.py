import torch.nn as nn
import torch
import torch.nn.functional as F


# TODO kausal oder nich-kausal entscheiden basierend auf padding

# input tensor is 2D -> if batch_size = 1, then has a shape of [1, number_of_windows, number_of_samples_per_window]
class S2T_CNN_frequency(nn.Module):
    def __init__(self,
                 vocab_size: int):
        super().__init__()

        ######## CONVOLUTIONAL LAYERS  ########
        self.cn1 = nn.Conv2d(in_channels=1,
                             out_channels=32,
                             kernel_size=5,
                             stride=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.cn2 = nn.Conv2d(in_channels=32,
                             out_channels=64,
                             kernel_size=5,
                             stride=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.cn3 = nn.Conv2d(in_channels=64,
                             out_channels=128,
                             kernel_size=5,
                             stride=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.cn4 = nn.Conv2d(in_channels=128,
                             out_channels=256,
                             kernel_size=5,
                             stride=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        ######## FULLY CONNECTED LAYERS  ########
        self.fc1 = nn.Linear(in_features=256*1*1,
                             out_features=200)
        self.fc2 = nn.Linear(in_features=200,
                             out_features=100,)
        self.fc3 = nn.Linear(in_features=100,
                             out_features=vocab_size)

        ######## DROPOUT LAYERS  ########
        self.drop1 = nn.Dropout(0.3)

    def forward(self,
                x: torch.Tensor):
        x = self.cn1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(input=x,
                         kernel_size=2,
                         stride=2)

        x = self.cn2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(input=x,
                         kernel_size=2,
                         stride=2)

        x = self.cn3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(input=x,
                         kernel_size=2,
                         stride=2)

        x = self.cn4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.gap(x)

        # flattening the input for the last linear layers
        x = torch.flatten(x,
                          start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.fc3(x)

        return x
