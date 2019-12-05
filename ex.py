import torch
from functools import partial
import numpy as np

import se3cnn
import se3cnn.SO3 as SO3
from se3cnn.point.operations import Convolution
from se3cnn.non_linearities import GatedBlock
from se3cnn.point.kernel import Kernel
from se3cnn.point.radial import CosineBasisModel
from se3cnn.non_linearities import rescaled_act



from spherical import SphericalTensor


import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go

torch.set_default_dtype(torch.float64)

square = torch.tensor(
    [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]
)
square -= square.mean(-2)
sx, sy = 0.5, 1.5
rectangle = square * torch.tensor([sx, sy, 0.])
rectangle -= rectangle.mean(-2)

N, _ = square.shape

markersize = 15


class Network(torch.nn.Module):
    def __init__(self, Rs, n_layers=3, sh=SO3.spherical_harmonics_xyz, max_radius=.5, number_of_basis=3, radial_layers=3):
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
        print(RadialModel)

        K = partial(Kernel, RadialModel=RadialModel, sh=sh)
        print(K)
        C = partial(Convolution, K)
        print(C)

        self.layers = torch.nn.ModuleList([
            GatedBlock(Rs, Rs, sp, rescaled_act.sigmoid, C)
            for _ in range(n_layers - 1)
        ])

        self.layers.append(
            Convolution(K, Rs, Rs)
        )

    def forward(self, input, geometry):
        output = input
        # print(geometry.shape) # 1, 4, 3
        batch, N, _ = geometry.shape
        # why do i divide by 2?
        for layer in self.layers:
            output = layer(output.div(N ** 0.5), geometry)
            print(output.shape)
        return output


L_max = 3
multiplicity = 1
Rs = [(multiplicity, L) for L in range(L_max + 1)]
print(Rs)

model = Network(Rs)
print(model)


params = model.parameters()
optimizer = torch.optim.Adam(params, 1e-3)
loss_fn = torch.nn.MSELoss()



z = torch.zeros(1, N, sum(2 * L + 1 for L in range(L_max + 1)))
z[:, :, 0] = 1.  # batch, point, channel
print("Input shape:", z.shape)

displacements = square - rectangle
N, _ = displacements.shape
projections = torch.stack([SphericalTensor.from_geometry(displacements[i], L_max).signal for i in range(N)])



iterations = 100
for i in range(iterations):
    output = model(z, rectangle.unsqueeze(0))
    loss = loss_fn(output, projections.unsqueeze(0))
    if i % 10 == 0:
        print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



output = model(z, rectangle.unsqueeze(0))


angles = torch.rand(3) * torch.tensor([np.pi, 2 * np.pi, np.pi])
rot = SO3.rot(*angles)
rot_rectangle = torch.einsum('xy,jy->jx', (rot, rectangle))
rot_square = torch.einsum('xy,jy->jx', (rot, square))
output = model(z, rot_rectangle.unsqueeze(0))





model = Network(Rs)

params = model.parameters()
optimizer = torch.optim.Adam(params, 1e-3)
loss_fn = torch.nn.MSELoss()




input = torch.zeros(1, N, sum(2 * L + 1 for L in range(L_max + 1)))
input[:, :, 0] = 1.  # batch, point, channel



displacements = rectangle - square
N, _ = displacements.shape
projections = torch.stack([SphericalTensor.from_geometry(displacements[i], L_max).signal for i in range(N)])




iterations = 100
for i in range(iterations):
    output = model(input, square.unsqueeze(0))
    loss = loss_fn(output, projections.unsqueeze(0))
    if i % 10 == 0:
        print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Fix
model = Network(Rs)
params = model.parameters()
optimizer = torch.optim.Adam(params, 1e-3)
loss_fn = torch.nn.MSELoss()



input = torch.zeros(1, N, sum(2 * L + 1 for L in range(L_max + 1)))
input[:, :, 0] = 1.  # batch, point, channel
# Breaking x and y symmetry with x^2 - y^2 component
input[:, :, 8] = 0.1  # x^2 - y^2

displacements = rectangle - square
N, _ = displacements.shape
projections = torch.stack([SphericalTensor.from_geometry(displacements[i], L_max).signal for i in range(N)])


iterations = 100
for i in range(iterations):
    output = model(input, square.unsqueeze(0))
    loss = loss_fn(output, projections.unsqueeze(0))
    if i % 10 == 0:
        print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


rows, cols = 1, 1
specs = [[{'is_3d': True} for i in range(cols)]
         for j in range(rows)]
fig = make_subplots(rows=rows, cols=cols, specs=specs)

L_max = 5
Rs = [(1, L) for L in range(L_max + 1)]
sum_Ls = sum(2 * L + 1 for mult, L in Rs)

# Random spherical tensor up to L_Max
signal = torch.zeros(sum_Ls)
signal[0] = 1
# Breaking x and y symmetry with x^2 - y^2
signal[8] = -0.1

sphten = SphericalTensor(signal, Rs)
