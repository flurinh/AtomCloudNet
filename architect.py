from Architectures.atomcloud import *
from Architectures.AtomCloudNet import *
from Architectures.cloud_utils import *
import os
from Processing.loader import *
import glob
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

path = 'data/hist'


n_batches = 1000
batch_size = 256
real_batch_size = 8
nepochs = 30

feats = ['prot', 'ph']


xyz = torch.randn((n_batches, batch_size, 3, 96))
features = torch.randint(1, 8, (n_batches, batch_size, 96))

feats = ['prot', 'ph']

data = xyz_loader(feats=feats, limit=1280, path=path + '/*.xyz')
data.plot_hist()
loader = DataLoader(data, batch_size=batch_size, shuffle=True)

model = AtomCloudNet()
criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
model.train()
opt.zero_grad()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number trainable parameters:", params)
print(model)


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


for e in trange(nepochs):
    for i, sample in enumerate(tqdm(loader), start = 1):
        xyz = sample[0].to(device)
        features = sample[1].to(device)
        target = sample[2].to(device).float()
        output = model(xyz, features)
        loss = criterion(output.float(), sample[2].float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.cpu().time())