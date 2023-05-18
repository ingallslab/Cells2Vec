import torch
import torch.nn as nn
import torch.optim as optim


class SiameseNetwork(nn.Module):
    def __init__(self, input_shape=(1, 1000), num_filters=32, kernel_size=(3, 3), output_dim=16):
        super(SiameseNetwork, self).__init__()
        self.input_shape = input_shape
        self.filters = num_filters
        self.kernel_size = kernel_size
        self.output_dim = output_dim

        # euclidean distance
        self.distance_layer = nn.PairwiseDistance(p=2)

        self.conv = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=1),
            # convert multi-dimensional tensor into a 1-dimensional tensor
            nn.Flatten(),
            nn.Linear(num_filters * input_shape[0] * input_shape[1], output_dim)
        )

    def forward_once(self, x):
        x = x.view(1, 1, 1, -1)
        x = self.conv(x)
        return x

    def forward(self, x1):
        encoding = self.forward_once(x1, x1.shape)
        distance = self.distance_layer(encoding, encoding)
        return distance

