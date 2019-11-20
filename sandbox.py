from Architectures.AtomCloudNet import *
from Processing.loader2 import *
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

path = 'data/QM9'



batch_size = 1
real_batch_size = 4
nepochs = 30

feats = ['prot', 'ph']

data = qm9_loader(feats=feats, limit=2000, path=path + '/*.xyz')
print("Total number of samples assembled:", data.__len__())
loader = DataLoader(data, batch_size=batch_size, shuffle=True)



model = AtomCloudFeaturePropagation().float()
print(model)

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

torch.autograd.set_detect_anomaly(True)
for e in trange(nepochs):
    for i, (xyz, Z, partial) in enumerate(tqdm(loader), start = 1):
        xyz = xyz.permute(0, 2, 1).to(device)
        feat = Z.view(batch_size, xyz.shape[2]).to(device)
        prediction = model(xyz, feat)
        #print(output.shape)
        new_loss = criterion(prediction, partial)
        print(output)
        print("loss:", new_loss.cpu().item())
        #print(output.shape)
        #print("target", sample[2].shape)
        # print(output.float(), sample[2].float())
        if i % real_batch_size == 1:
            loss = new_loss
        else:
            loss = loss + new_loss
        if i % real_batch_size == 0:
            loss = loss / real_batch_size
            loss.backward()
            opt.step()
            print("======================================================================")
            print()
            print("output:", output)
            print("target:", sample[2])
            print("")
            print("loss:", torch.sqrt(loss).cpu().item())
            print()
            print("======================================================================")
            opt.zero_grad()
    torch.save(model, 'model.pt')
