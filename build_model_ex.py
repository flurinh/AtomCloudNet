import torch
import se3cnn
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

# Inside se3cnn.point.radial.CosineBasisModel
import math
import matplotlib.pyplot as plt

max_radius = 3.0
number_of_basis = 3
radii = torch.linspace(0, max_radius, steps=number_of_basis)
step = radii[1] - radii[0]
basis = lambda x: x.div(step).add(1).relu().sub(2).neg().relu().add(1).mul(math.pi / 2).cos().pow(2)

x = torch.linspace(-max_radius, max_radius, 1000)
plt.plot(x, basis(x))


# se3cnn has operations for point sets and for 3D images. We will be using points.
import se3cnn.point
import se3cnn.point.radial
from functools import partial
from se3cnn.non_linearities import rescaled_act

# We are going to define RadialModel by specifying every single argument
# of CosineBasisModel EXCEPT out_dim which will be passed later
radial_layers = 2
sp = rescaled_act.Softplus(beta=5)
RadialModel = partial(se3cnn.point.radial.CosineBasisModel, max_radius=max_radius,
                      number_of_basis=number_of_basis, h=100,
                      L=radial_layers, act=sp)



import se3cnn.point.kernel

sh = None
K = partial(se3cnn.point.kernel.Kernel, RadialModel=RadialModel, sh=sh)



import se3cnn.point.operations

# If we wish to pass the convolution to a layer definition
C = partial(se3cnn.point.operations.Convolution, K)

# Or alternatively, if we want to use the convolution directly,
# we need to specify the `Rs` of the input and output
Rs_in = [(2, 0)]
Rs_out = [(4, 0), (4, 1), (4, 2)]
convolution = se3cnn.point.operations.Convolution(K, Rs_in, Rs_out)


import se3cnn.non_linearities as nl
from se3cnn.non_linearities import rescaled_act

gated_block = nl.gated_block.GatedBlock(Rs_in, Rs_out, sp, rescaled_act.sigmoid, C)

dimensionalities = [2 * L + 1 for mult, L in Rs_out for _ in range(mult)]
norm_activation = nl.norm_activation.NormActivation(dimensionalities, rescaled_act.sigmoid, rescaled_act.sigmoid)

print(K)
print(C)
print(gated_block)