from Architectures.AtomCloudNet import *
from Processing.loader2 import *

from torch.utils.data import DataLoader

from tqdm import tqdm, trange


path = 'data/QM9'
batch_size = 1
real_batch_size = 1
nepochs = 30

feats = ['prot', 'ph']

data = qm9_loader(feats=feats, limit=240, path=path + '/*.xyz')
print("Total number of samples assembled:", data.__len__())
loader = DataLoader(data, batch_size=batch_size, shuffle=True)


model = AtomCloudFeaturePropagation().float()
print(model)

criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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

torch.autograd.set_detect_anomaly(True)
pbar = tqdm(loader)
for e in trange(nepochs):
    loss_ = np.inf
    tot_loss = 0
    avg_loss = loss_
    prediction = [np.inf]
    for i, (xyz, Z, prot_ids, partial, urt) in enumerate(pbar, start=1):
        xyz = xyz.permute(0, 2, 1).to(device)
        feat = Z.view(xyz.shape[0], xyz.shape[2]).to(device)
        prediction = model(xyz, feat)
        loss = criterion(prediction, urt.to(device))
        loss_ = torch.sqrt(loss).cpu().item() * 600
        tot_loss += loss_
        avg_loss = tot_loss / i
        pbar.set_description("epoch-avg-loss::{} --------  loss::{}  --------  prediction::{}  --------  target::{}  "
                             .format(avg_loss, loss_, prediction[0], urt[0]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(model, 'model2.pt')
