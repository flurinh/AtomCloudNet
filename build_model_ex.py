import torch
import se3cnn
import math
import matplotlib.pyplot as plt

import se3cnn.point
import se3cnn.point.radial
from functools import partial
from se3cnn import SO3
from se3cnn.non_linearities import rescaled_act

import numpy as np

import se3cnn.point.kernel
import se3cnn.point.operations


import se3cnn.non_linearities as nl
from se3cnn.non_linearities import rescaled_act

from spherical import *

import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go

torch.set_default_dtype(torch.float64)


Rs = [(2, 0)] # Two (2) scalar (L=0) channels: hydrogen and carbon

# 3D coordinates of the atoms of the molecule
C_geo = torch.tensor(
    [[ 0.     ,  1.40272,  0.     ],
     [-1.21479,  0.70136,  0.     ],
     [-1.21479, -0.70136,  0.     ],
     [ 0.     , -1.40272,  0.     ],
     [ 1.21479, -0.70136,  0.     ],
     [ 1.21479,  0.70136,  0.     ]]
)
H_geo = torch.tensor(
    [[ 0.     ,  2.49029,  0.     ],
     [-2.15666,  1.24515,  0.     ],
     [-2.15666, -1.24515,  0.     ],
     [ 0.     , -2.49029,  0.     ],
     [ 2.15666, -1.24515,  0.     ],
     [ 2.15666,  1.24515,  0.     ]]
)
geometry = torch.cat([C_geo, H_geo], axis=-2)

# and features on each atom
C_input = torch.tensor([[0., 1.] for i in range(C_geo.shape[-2])])
H_input = torch.tensor([[1., 0.] for i in range(H_geo.shape[-2])])
input = torch.cat([C_input, H_input])

# print(geometry.shape)
# print(input.shape)
# Inside se3cnn.point.radial.CosineBasisModel

max_radius = 1
number_of_basis = 3
radii = torch.linspace(0, max_radius, steps=number_of_basis)
step = radii[1] - radii[0]
basis = lambda x: x.div(step).add(1).relu().sub(2).neg().relu().add(1).mul(math.pi / 2).cos().pow(2)

x = torch.linspace(-max_radius, max_radius, 1000)
# plt.plot(x, basis(x))
# plt.show()

# se3cnn has operations for point sets and for 3D images. We will be using points.
# We are going to define RadialModel by specifying every single argument
# of CosineBasisModel EXCEPT out_dim which will be passed later
radial_layers = 2
sp = rescaled_act.Softplus(beta=5)
RadialModel = partial(se3cnn.point.radial.CosineBasisModel,
                      max_radius=max_radius,
                      number_of_basis=number_of_basis,
                      h=100,
                      L=radial_layers,
                      act=sp)




sh = SO3.spherical_harmonics_xyz
K = partial(se3cnn.point.kernel.Kernel, RadialModel=RadialModel, sh=sh)



# If we wish to pass the convolution to a layer definition
C = partial(se3cnn.point.operations.Convolution, K)

# Or alternatively, if we want to use the convolution directly,
# we need to specify the `Rs` of the input and output
Rs_in = [(2, 0)]
Rs_out = [(4, 0), (4, 1), (4, 2), (4, 3)]
convolution = se3cnn.point.operations.Convolution(K, Rs_in, Rs_out)
neighborconv = se3cnn.point.operations.NeighborsConvolution(K, Rs_in, Rs_out, radius=1.8)
print(neighborconv.radius)


gated_block = nl.gated_block.GatedBlock(Rs_in, Rs_out, sp, rescaled_act.sigmoid, C)

dimensionalities = [2 * L + 1 for mult, L in Rs_out for _ in range(mult)]
norm_activation = nl.norm_activation.NormActivation(dimensionalities, rescaled_act.sigmoid, rescaled_act.sigmoid)


model_parameters = filter(lambda p: p.requires_grad, gated_block.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of parameters in conv:", params)
transformed = neighborconv(input.unsqueeze(0), geometry.unsqueeze(0))


def plot_neighborhood(geometry, features, rs, label):
    N, _ = geometry.shape
    rows, cols = 1, 1
    specs = [[{'is_3d': True} for i in range(cols)]
             for j in range(rows)]
    fig = make_subplots(rows=rows, cols=cols, specs=specs)
    fig.add_trace(go.Scatter3d(x=geometry[:, 0], y=geometry[:, 1], z=geometry[:, 2], mode="markers", name=label))
    for i in range(N):
        sph_ten_input = features[0][i].detach()
        trace = SphericalTensor(sph_ten_input, rs).plot(center=geometry[i])
        trace.showscale = False
        fig.add_trace(trace, 1, 1)
    return fig

print("geometry:", geometry.shape)
print("input:", input.shape)
print("Rs input:", Rs_in)
print("transformed:", transformed.shape)
print("Rs out:", Rs_out)

fig = plot_neighborhood(geometry, input.unsqueeze(0), Rs_in, "Input")
fig.update_layout(scene_aspectmode='data')
fig.show()


fig = plot_neighborhood(geometry, transformed, Rs_out, "Transformed")
fig.update_layout(scene_aspectmode='data')
fig.show()

"""
class Network(torch.nn.Module):
    def __init__(self, Rs, n_layers=3, sh=SO3.spherical_harmonics_xyz, max_radius=3.0, number_of_basis=3, radial_layers=3):
        super().__init__()
        self.Rs = Rs
        self.n_layers = n_layers
        self.L_max = max(L for m, L in Rs)

        sp = rescaled_act.Softplus(beta=5)

        Rs_geo = [(1, l) for l in range(self.L_max + 1)]
        Rs_centers = [(1, 0), (1, 1)]

        RadialModel = partial(CosineBasisModel, max_radius=max_radius,
                              number_of_basis=number_of_basis, h=100,
                              L=radial_layers, act=sp)

        K = partial(Kernel, RadialModel=RadialModel, sh=sh)
        C = partial(Convolution, K)

        self.layers = torch.nn.ModuleList([
            GatedBlock(Rs, Rs, sp, rescaled_act.sigmoid, C)
            for i in range(n_layers - 1)
        ])

        self.layers.append(
            Convolution(K, Rs, Rs)
        )

    def forward(self, input, geometry):
        output = input
        # print(geometry.shape) # 1, 4, 3
        batch, N, _ = geometry.shape
        for layer in self.layers:
            output = layer(output.div(N ** 0.5), geometry)
        return output
"""