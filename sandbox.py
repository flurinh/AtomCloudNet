from Architectures.point_util import *
from Architectures.pointconv2 import *
import os
from Processing.loader import *
import glob
from torch.utils.data import DataLoader

"""
input = torch.randn((8, 3, 128))
features = torch.randn((8, 3, 128))
model = PointNet2ClsSsg()
print(model)
output = model(input, features)
print(output)
"""

path = 'data/hist'



data = xyz_loader(limit=1280, path=path + '/*.xyz')


batch_size = 1
loader = DataLoader(data, batch_size=batch_size, shuffle=True)
model = PointNet2ClsSsg().double()

criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
opt.zero_grad()
real_batch_size = 128
nepochs = 10
for e in range(nepochs):
    for i, sample in enumerate(tqdm(loader)):
        xyz = sample[0].permute(0, 2, 1)
        feat = sample[1].view(batch_size, 1, -1)
        output = model(xyz, feat)
        if i % real_batch_size == 1 or i == 0:
            loss = criterion(sample[2], output)
        elif i % real_batch_size == 0:
            loss += criterion(sample[2], output)
            loss = loss / real_batch_size
            loss.backward()
            opt.step()
            print(loss.item())
            opt.zero_grad()
