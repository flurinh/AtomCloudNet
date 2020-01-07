from Architectures.AtomCloudNet import *
from Processing.loader2 import *

from torch.utils.data import DataLoader

from tqdm import tqdm, trange


class ACN:
    def __init__(self):
        self.verbose = 1
        self.run_id = 1
        self.path = 'data/QM9'
        self.save_path = 'models/model_' + str(self.run_id).zfill(5)
        self.batch_size = 16
        self.real_batch_size = 1
        self.nepochs = 30
        self.ngpus = 0
        feats = ['prot', 'ph']

        data = qm9_loader(feats=feats, limit=1000, path=self.path + '/*.xyz')
        print("\nTotal number of samples assembled:", data.__len__())
        self.loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        self.criterion = nn.MSELoss()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Using device:", self.device)

    def train_molecular_model(self):
        history = []

        model = se3AtomCloudNet(device=self.device).float()
        model.train()

        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        opt.zero_grad()
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        if self.verbose > 0:
            print(model)
            print("Number trainable parameters:", params)

        if torch.cuda.device_count() > 1:
            self.ngpus = torch.cuda.device_count()
            print("Let's use" + str(self.ngpus) + "GPUs!")
            model = torch.nn.DataParallel(model)
        model.to(self.device)

        torch.autograd.set_detect_anomaly(True)
        pbar = tqdm(self.loader)
        for _ in trange(self.nepochs):
            tot_loss = 0
            for i, (xyz, Z, prot_ids, partial, urt) in enumerate(pbar, start=1):
                xyz = xyz.to(self.device)
                feat = Z.view(xyz.shape[0], xyz.shape[1]).to(self.device)
                prediction = model(xyz, feat)
                loss = self.criterion(prediction, urt.to(self.device))
                loss_ = torch.sqrt(loss).cpu().item() * 600
                tot_loss += loss_
                avg_loss = tot_loss / i
                pbar.set_description("epoch-avg-loss::{}  "
                                     "--------  loss::{}  "
                                     "--------  prediction::{}  "
                                     "--------  target::{}  "
                                     .format(avg_loss, loss_, prediction[0].cpu().detach().numpy().round(3),
                                             urt[0].cpu().detach().numpy().round(3)))
                opt.zero_grad()
                loss.backward()
                opt.step()
            # torch.save(model, self.save_path + '.pt')
        return model, history

    def train_atomic_model(self):
        history = []
        model = AtomCloudFeaturePropagation(device=self.device).float()
        model.train()

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.zero_grad()
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        if self.verbose > 0:
            print(model)
            print("Number trainable parameters:", params)

        if torch.cuda.device_count() > 1:
            self.ngpus = torch.cuda.device_count()
            print("Let's use" + str(self.ngpus) + "GPUs!")
            model = torch.nn.DataParallel(model)
        model.to(self.device)

        torch.autograd.set_detect_anomaly(True)
        pbar = tqdm(self.loader)
        for _ in trange(self.nepochs):
            tot_loss = 0
            for i, (xyz, Z, prot_ids, partial, urt) in enumerate(pbar, start=1):
                opt.zero_grad()
                xyz = xyz.permute(0, 2, 1).to(self.device)
                feat = Z.view(xyz.shape[0], xyz.shape[2]).to(self.device)
                prediction = model(feat, xyz)
                loss = self.criterion(prediction, urt.to(self.device))
                loss_ = torch.sqrt(loss).cpu().item() * 600
                tot_loss += loss_
                avg_loss = tot_loss / i
                pbar.set_description("epoch-avg-loss::{}  "
                                     "--------  loss::{}  "
                                     "--------  prediction::{}  "
                                     "--------  target::{}  "
                                     .format(avg_loss, loss_, prediction[0].cpu().detach().numpy(), urt[0].cpu().detach().numpy()))
                loss.backward()
                opt.step()
        return model, history


net = ACN()
print("Starting training...")
net.train_molecular_model()
