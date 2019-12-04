import torch
import torch.nn as nn
import torch.nn.functional as F

from se3cnn.non_linearities.gated_block import GatedBlock


class AvgSpacial(torch.nn.Module):
    def forward(self, inp):  # pylint: disable=W
        # inp [batch, features, x, y, z]
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)  # [batch, features]


class Model(torch.nn.Module):
    input_size = 77

    def __init__(self, n_in, n_out):
        super().__init__()

        self.int_repr = None

        features = [
            (n_in, ), # 77
            (12, 4, 1),
            (24, 8, 2),
            (24, 8, 2),
            (24, 8, 2),
            (n_out, ),
        ]

        block_params = [
            {'activation': (F.relu, torch.sigmoid), 'stride': 2},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': None},
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [
            GatedBlock(features[i], features[i + 1], **block_params[i])
            for i in range(len(block_params))
        ]

        for p in blocks[-1].parameters():
            nn.init.zeros_(p)

        self.sequence = torch.nn.Sequential(
            *blocks,
            AvgSpacial(),
        )


    def forward(self, x):  # pylint: disable=W
        '''
        :param x: [batch, features, x, y, z]
        '''
        return self.sequence(x)