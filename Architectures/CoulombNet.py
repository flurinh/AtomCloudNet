import torch
import torch.nn as nn
import torch.nn.functional as F

class CoulombNet(nn.Module):
    """
    Simple feedforward architecture neural network. Baseline module.
    """

    def __init__(self, in_shape, layers, dropout):
        super(CoulombNet, self).__init__()
        self.last_layer_out = None
        self.coul_blocks = nn.ModuleList()
        in_channel = in_shape
        for _ in range(len(layers)):
            self.coul_blocks.append(nn.Linear(in_channel, layers[_]))
            self.coul_blocks.append(nn.BatchNorm1d(layers[_])) # TODO: if batch larger than one, always normalized
            self.coul_blocks.append(nn.Dropout(dropout))    # TODO: dropout
            self.coul_blocks.append(nn.Sigmoid())       # TODO: sigmoid if labels btw 0-1   0-lim(infinite) Relu, -1-1 Tanh
            in_channel = layers[_]
        self.final = nn.Linear(in_channel, 1)
        self.fact = torch.nn.Sigmoid()  # self.fact = nn.Sigmoid() # TODO: final layer sigmoid

    def forward(self, x):
        for _, block in enumerate(self.coul_blocks):
            x = block(x)
        self.last_layer_out = x.clone().detach().numpy()

        """for _, block in enumerate(self.coul_blocks):
                    x = F.relu(block(x))"""

        return self.fact(self.final(x))