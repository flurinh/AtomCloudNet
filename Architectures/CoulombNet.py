import torch
import torch.nn as nn


class CoulombNet(nn.Module):
    """
    Simple feedforward architecture neural network. Baseline module.
    """
    def __init__(self, in_shape, layers=[1024, 512, 256]):
        super(CoulombNet, self).__init__()
        self.coul_blocks = nn.ModuleList()
        in_channel = in_shape
        for _ in range(len(layers)):
            self.coul_blocks.append(nn.Linear(in_channel, layers[_]))
            self.coul_blocks.append(nn.ReLU(layers[_]))
            in_channel = layers[_]
        self.final = nn.Linear(in_channel, 1)
        self.fact = nn.Sigmoid()

    def forward(self, x):
        for _, block in enumerate(self.coul_blocks):
            x = block(x)
        return self.fact(self.final(x))