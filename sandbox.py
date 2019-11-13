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



batch_size = 1
real_batch_size = 128
nepochs = 30

feats = ['prot']

data = xyz_loader(feats=feats, limit=12800, path=path + '/*.xyz')
loader = DataLoader(data, batch_size=batch_size, shuffle=True)
model = PointNet2ClsSsg(nfeats=len(feats)).double()

criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=5e-5)
model.train()
opt.zero_grad()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number trainable parameters:", params)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)
multi_gpu = None
if multi_gpu is not None:
    device_ids = [int(x) for x in multi_gpu.split(',')]
    torch.backends.cudnn.benchmark = True
    model.cuda(device_ids[0])
    model = torch.nn.DataParallel(model, device_ids=device_ids)
else:
    model.to(device)


for e in range(nepochs):
    for i, sample in enumerate(tqdm(loader), start = 1):
        xyz = sample[0].permute(0, 2, 1).to(device)
        feat = sample[1].view(batch_size, -1, xyz.shape[2]).to(device)
        output = model(xyz, feat)
        new_loss = criterion(output.float(), sample[2].float())
        # print(output.float(), sample[2].float())
        if i % real_batch_size == 1:
            loss = new_loss
        else:
            loss = loss + new_loss
        if i % real_batch_size == 0:
            loss = loss / real_batch_size
            loss.backward()
            opt.step()
            print("loss:", torch.sqrt(loss).cpu().item() * 50)
            opt.zero_grad()
    torch.save(model, 'model.pt')
