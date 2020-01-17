import os
os.system('tensorboard --logdir runs/run_130 --port 6006')
# scp -r /Users/modlab/PycharmProjects/Protos/data/QM9 hidberf@login.leonhard.ethz.ch:/cluster/home/hidberf/Protos/data
# for i in range(48):
#    os.system('bkill '+str(3797019+i))