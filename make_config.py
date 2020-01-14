from configparser import ConfigParser
import argparse
import os

# self.['SETTING']['layers'] = ast.literal_eval(config.get("section", "option"))


class Config:
    def __init__(self, setting):
        if setting == 0:
            print("CoulombNet configuration generator... ")
            path = "config/config"
            config_file = path+".ini"
            self.parser = ConfigParser()
            self.parser.read(config_file)
            self.idx = 0
            self.parser['SETTING']['model'] = 'NN'

            lr = [0.1, 0.005, 0.0001]
            momentum = [0, 0.1, 0.5]
            dropout = [0, 0.1, 0.2, 0.3]
            patience = [10, 25, 50]
            pfactor = [0.5]
            epochs = [50, 200, 500]
            batchsize = [8, 64, 256]
            architectures = [[1024, 1024, 512],
                            [50, 50, 50, 50, 50, 50],
                            [1024, 512, 128, 32],
                            [512, 256, 64, 64, 32, 32]]

            for l in lr:
                for m in momentum:
                    for d in dropout:
                        for p in patience:
                            for pf in pfactor:
                                for e in epochs:
                                    for b in batchsize:
                                        for a in architectures:
                                            self.idx += 1
                                            new_config = path+'_'+str(setting+self.idx).zfill(5)+'.ini'
                                            self.parser['SETTING']['lr'] = str(l)
                                            self.parser['SETTING']['momentum'] = str(m)
                                            self.parser['SETTING']['dropout'] = str(d)
                                            self.parser['SETTING']['patience'] = str(p)
                                            self.parser['SETTING']['pfactor'] = str(pf)
                                            self.parser['SETTING']['layers'] = str(a)
                                            self.parser['SETTING']['epochs'] = str(e)
                                            self.parser['SETTING']['batchsize'] = str(b)
                                            if os.path.isfile(new_config):
                                                os.remove(new_config)
                                            with open(new_config, "w") as f:
                                                self.parser.write(f)

        if setting == 1:
            print("AtomCloudNet configuration generator.... WRUMM WRUMM")
            path = "config_ACN/config"
            config_file = path + ".ini"
            self.parser = ConfigParser()
            self.parser.read(config_file)
            self.idx = 1200

            model = [1, 2, 3]
            lr = [0.0005, 0.0001]
            epochs = [150]
            batchsize = [16]
            neighborradius = [3, 4, 5]
            nclouds = [1, 2]
            clouddim = [4, 12]
            cloudord = [3]
            resblocks = [2]
            nffl = [2]
            ffl1size = [128]
            emb_dim = [16]

            # Todo: optimize: nffl=1, ffl1size=128, emb_dim=32
            for m in model:
                for l in lr:
                    for e in epochs:
                        for b in batchsize:
                            for n in neighborradius:
                                for nc in nclouds:
                                    for cd in clouddim:
                                        for co in cloudord:
                                            for rb in resblocks:
                                                for nf in nffl:
                                                    for ff in ffl1size:
                                                        for ed in emb_dim:
                                                            self.idx += 1
                                                            new_config = path + '_' + str(self.idx).zfill(5) + '.ini'
                                                            self.parser['SETTING']['model'] = str(m)
                                                            self.parser['SETTING']['lr'] = str(l)
                                                            self.parser['SETTING']['neighborradius'] = str(n)
                                                            self.parser['SETTING']['nclouds'] = str(nc)
                                                            self.parser['SETTING']['clouddim'] = str(cd)
                                                            self.parser['SETTING']['resblocks'] = str(rb)
                                                            self.parser['SETTING']['nffl'] = str(nf)
                                                            self.parser['SETTING']['ffl1size'] = str(ff)
                                                            self.parser['SETTING']['emb_dim'] = str(ed)
                                                            self.parser['SETTING']['epochs'] = str(e)
                                                            self.parser['SETTING']['batchsize'] = str(b)
                                                            self.parser['SETTING']['cloudord'] = str(co)
                                                            if os.path.isfile(new_config):
                                                                os.remove(new_config)
                                                            with open(new_config, "w") as f:
                                                                self.parser.write(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify setting (generates all corresponding .ini files).')
    parser.add_argument('--setting', type=int, default=1)
    args = parser.parse_args()
    Config(setting=args.setting)




