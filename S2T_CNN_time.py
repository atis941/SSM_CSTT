import torch.nn as nn
import torch
import torch.nn.functional as F


# TODO kausal oder nich-kausal entscheiden basierend auf padding

# input tensor is 2D -> if batch_size = 1, then has a shape of [1, number_of_windows, number_of_samples_per_window]
class S2T_CNN_time_V1(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 kernel_size: int,
                 padding: int,
                 padding_type: str,
                 stride: int,
                 in_channels: list,
                 out_channels: list,
                 lin_dropout: int
                 ):
        super().__init__()

        ######## CONVOLUTIONAL LAYERS  ########
        self.cn1 = nn.Conv2d(in_channels=in_channels[0],
                             out_channels=out_channels[0],
                             kernel_size=kernel_size,
                             padding=padding,
                             stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels[0])

        self.cn2 = nn.Conv2d(in_channels=in_channels[1],
                             out_channels=out_channels[1],
                             kernel_size=kernel_size,
                             padding=padding,
                             stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels[1])

        self.cn3 = nn.Conv2d(in_channels=in_channels[2],
                             out_channels=out_channels[2],
                             kernel_size=kernel_size,
                             padding=padding,
                             stride=stride)
        self.bn3 = nn.BatchNorm2d(out_channels[2])

        self.cn4 = nn.Conv2d(in_channels=in_channels[3],
                             out_channels=out_channels[3],
                             kernel_size=kernel_size,
                             padding=padding,
                             stride=stride)
        self.bn4 = nn.BatchNorm2d(out_channels[3])

        ######## FULLY CONNECTED LAYERS  ########
        self.fc1 = nn.Linear(in_features=out_channels[-1],
                             out_features=vocab_size)

        ######## DROPOUT LAYERS  ########
        self.drop1 = nn.Dropout(lin_dropout)

    def forward(self,
                x: torch.Tensor):

        # adding an additional dimension for the CHANNELS to the input before passing it to the first Conv layer
        # necessary because it expacts an input in the shape of [batch, CHANNELS, height=windows, width=mel_freq_bin]
        x = x.unsqueeze(1)

        x = self.cn1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(input=x,
                         # kernel size shouldnt be stretched across the time dimension
                         kernel_size=(1, 2),
                         stride=(1, 2))  # the pooling should be only applied across the frequency dimension

        x = self.cn2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(input=x,
                         kernel_size=(1, 2),
                         stride=(1, 2))

        x = self.cn3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(input=x,
                         kernel_size=(1, 2),
                         stride=(1, 2))

        x = self.cn4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(input=x,
                         kernel_size=(1, 2),
                         stride=(1, 2))

        # flattening the input for the last linear layers
        # calculate a mean across the freq dimension: [batch, output_channels, time_dim, freq_dim] -> [batch, output_channels, time_dim]
        x = x.mean(dim=-1)

        # transpose so that the temporal (time) dimension is in the middle and the basically the extracted feature channels will be the frequency bins
        x = x.transpose(1, 2)

        # pass the input to the classification layer
        x = self.fc1(x)

        return x
