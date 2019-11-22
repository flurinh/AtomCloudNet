from Architectures.AtomCloudNet import *
from Processing.loader2 import *

from torch.utils.data import DataLoader

from tqdm import tqdm, trange


path = 'data/QM9'
batch_size = 16
real_batch_size = 1
nepochs = 30

feats = ['prot', 'ph']

data = qm9_loader(feats=feats, limit=24000, path=path + '/*.xyz')
print("Total number of samples assembled:", data.__len__())
loader = DataLoader(data, batch_size=batch_size, shuffle=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)


model = AtomCloudFeaturePropagation(device=device).float()
print(model)

criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
opt.zero_grad()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number trainable parameters:", params)

if torch.cuda.device_count() > 1:
    ngpus = torch.cuda.device_count()
    # print("Let's use" + str(ngpus) + "GPUs!")
    model = torch.nn.DataParallel(model)
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
        pbar.set_description("ngpus::{}  "
                             "--------  epoch-avg-loss::{}  "
                             "--------  loss::{}  "
                             "--------  prediction::{}  "
                             "--------  target::{}  "
                             .format(ngpus, avg_loss, loss_, prediction[0], urt[0]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(model, 'model2.pt')
