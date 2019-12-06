
import torch
import numpy as np
import se3cnn.SO3 as SO3
from spherical import SphericalTensor # a small Signal class written for ease of handling Spherical Tensors
import plotly
from plotly.subplots import make_subplots

import json
from functools import partial

import torch
import random
from se3cnn.non_linearities import GatedBlock
from se3cnn.non_linearities.rescaled_act import relu, sigmoid
from se3cnn.point.kernel import Kernel
from se3cnn.point.operations import PeriodicConvolution
from se3cnn.point.radial import CosineBasisModel



rows, cols = 1, 1
specs = [[{'is_3d': True} for i in range(cols)]
         for j in range(rows)]
fig = make_subplots(rows=rows, cols=cols, specs=specs)

L_max = 8

tetra_coords = torch.tensor( # The easiest way to construct a tetrahedron is using opposite corners of a box
    [[.5, .5, .5], [0,1,0], [1.5, 1.5, 1.5]]
)



Rs = [(1, L) for L in range(L_max + 1)]
sum_Ls = sum(2 * L + 1 for mult, L in Rs)

# Random spherical tensor up to L_Max
rand_sph_tensor = torch.randn(sum_Ls)

sphten = SphericalTensor(rand_sph_tensor, Rs)


tetra_coords -= tetra_coords.mean(-2)

fig = make_subplots(rows=rows, cols=cols, specs=specs)

sphten = SphericalTensor.from_geometry(tetra_coords, L_max)
print(sphten.signal)
trace = sphten.plot(relu=False, n=60)
fig.add_trace(trace, row=1, col=1)
fig.show()



class AvgSpacial(torch.nn.Module):
    def forward(self, features):
        return features.mean(1)


class Network(torch.nn.Module):
    def __init__(self, num_classes, L = 2):
        super().__init__()

        representations = [(1,), (4, 4, 4, 4), (4, 4, 4, 4), (4, 4, 4, 4), (64,)]
        representations = [[(mul, l) for l, mul in enumerate(rs)] for rs in representations]
        print(representations)
        R = partial(CosineBasisModel, max_radius=2, number_of_basis=10, h=100, L=L, act=relu)
        K = partial(Kernel, RadialModel=R)
        C = partial(PeriodicConvolution, Kernel=K, max_radius=2)

        self.firstlayers = torch.nn.ModuleList([
            GatedBlock(Rs_in, Rs_out, relu, sigmoid, C)
            for Rs_in, Rs_out in zip(representations, representations[1:])
        ])
        self.lastlayers = torch.nn.Sequential(AvgSpacial(), torch.nn.Linear(64, num_classes))

    def forward(self, structure):
        num_neighbors = 4
        N = num_neighbors
        p = next(self.parameters())
        geometry = torch.stack([p.new_tensor(s.coords) for s in structure.sites])
        features = p.new_ones(1, len(geometry), 1)
        geometry = geometry.unsqueeze(0)

        for i, m in enumerate(self.firstlayers):
            assert torch.isfinite(features).all(), i
            features = m(features.div(N ** 0.5), geometry, structure.lattice)

        return self.lastlayers(features).squeeze(0)

model = Network(5)
print(model)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)