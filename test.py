from Processing.loader import *
from Protos import *
from Architectures.CoulombNet import *

import torch.nn as nn

from tqdm import tqdm, trange
import numpy as np

from torch.utils.data import DataLoader


nepochs = 500
limit_atoms = 150

load_cv = coul_loader(limit=51200, limit_atoms=limit_atoms)
training = DataLoader(load_cv, batch_size=512, shuffle=True)


model = CoulombNet(limit_atoms, layers=[1024, 512, 256]).float()

criterion = nn.MSELoss()

opt = torch.optim.Adam(model.parameters(), lr=1e-2) #lr=1e-3

opt.zero_grad()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number trainable parameters:", params)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)
model.to(device)
print(model)
model.train()
for epoch in range(nepochs):
    epoch_loss = []
    for i, batch in enumerate(tqdm(training)):
        opt.zero_grad()
        output = model(batch[0].to(device).float())
        loss = criterion(output, batch[1].float())
        loss.backward()
        opt.step()
        epoch_loss.append(np.sqrt(loss.cpu().item()) * 50)
        print(output[0].item(), batch[1][0].item())
    print(sum(epoch_loss) / len(epoch_loss))