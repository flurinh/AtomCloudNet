import os
os.system('tensorboard --logdir runs/run_12 --port 6006')
# scp -r hidberf@login.leonhard.ethz.ch:/cluster/home/hidberf/Protos/runs "C:/Users/Flurin Hidber/PycharmProjects/Protos"
# for i in range(48):
#    os.system('bkill '+str(3797019+i))