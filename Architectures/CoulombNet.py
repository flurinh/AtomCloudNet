import torch
import torch.nn as nn

class CoulombNet(nn.Module):
    def __init__(self, in_shape, layers=[1024, 512, 256, 1]):
        super(CoulombNet, self).__init__()
        self.coul_blocks = nn.ModuleList()
        in_channel = in_shape
        for _ in range(layers):
            self.coul_blocks.append(nn.Dense(in_channel, layers[_]))
            self.coul_blocks.append(nn.ReLU(layers[_]))
            in_channel = layers[_]

    def forward(self, x):
        return (self.coul_blocks(x))