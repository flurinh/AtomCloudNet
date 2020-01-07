from configparser import ConfigParser
import argparse
import os

# self.['SETTING']['layers'] = ast.literal_eval(config.get("section", "option"))

class Config:
    def __init__(self, setting):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify setting (generates all corresponding .ini files).')
    parser.add_argument('--setting', type=int, default=0, help='Please specify setting:\n'
                                                               '   0 : Testing purposes\n')
    args = parser.parse_args()
    Config(setting=args.setting)