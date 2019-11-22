from Architectures.AtomCloudNet import *
from Processing.loader2 import *

from torch.utils.data import DataLoader

from tqdm import tqdm, trange


path = 'data/QM9'
batch_size = 8
real_batch_size = 1
nepochs = 30

feats = ['prot', 'ph']

data = qm9_loader(feats=feats, limit=24000, path=path + '/*.xyz')
print("Total number of samples assembled:", data.__len__())
loader = DataLoader(data, batch_size=batch_size, shuffle=True)


model = AtomCloudFeaturePropagation().float()
print(model)

criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
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

for e in trange(nepochs):
    for i, (xyz, Z, prot_ids, partial, urt) in enumerate(tqdm(loader), start=1):
        xyz = xyz.permute(0, 2, 1).to(device)
        feat = Z.view(xyz.shape[0], xyz.shape[2]).to(device)
        prediction = model(xyz, feat)
        # print(output.shape)
        # loss = criterion(prediction, partial)
        # loss = torch.sum(torch.abs(prediction-partial)) / prediction.shape[0]
        loss = criterion(prediction, urt.to(device))
        print("prediction", prediction[:3])
        print("target", urt[:3])
        print("loss:", torch.sqrt(loss).cpu().item() * 600)
        # print(output.shape)
        # print("target", sample[2].shape)
        # print(output.float(), sample[2].float())
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(model, 'model2.pt')
