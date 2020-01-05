import torch
from Architectures.AtomCloudNet import *
from spherical import *

torch.set_default_dtype(torch.float64)


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

# and Z on each atom
C_input = torch.LongTensor([1 for i in range(C_geo.shape[-2])])
H_input = torch.LongTensor([0 for i in range(H_geo.shape[-2])])
features = torch.LongTensor(torch.cat([C_input, H_input]))
xyz = geometry


model = se3AtomCloudNet()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

print(features.size())
print(xyz.size())

new_features, collation = model(features.unsqueeze(0), xyz.unsqueeze(0))
Rs_in = [(1, 0)]
Rs_out = [(8, 0), (8, 1), (8, 2), (8, 3)]

feat_plot = features.unsqueeze(0).unsqueeze(2)
print(feat_plot.size())
fig = plot_neighborhood(xyz, feat_plot, Rs_in, "Input")
fig.update_layout(scene_aspectmode='data')
fig.show()

transformed = new_features[:, :, :256]
print(transformed.size())
fig = plot_neighborhood(xyz, transformed, Rs_out, "Transformed")
fig.update_layout(scene_aspectmode='data')
fig.show()
