import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=True, type=int)
args, unparsed = parser.parse_known_args()
assert args.gpu in [0,1,2,3]

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

experiment_calls = [
    'python cath.py --data-filename cath_10arch_ca.npz --model ResNet34 --training-epochs 100 --batch-size 16 --batchsize-multiplier 1 --randomize-orientation --kernel-size 3 --initial_lr=0.0005 --lr_decay_start=40 --burnin-epochs 40 --lr_decay_base=.94 --downsample-by-pooling --p-drop-conv 0.01 --report-frequency 1 --lamb_conv_weight_L1 1e-7 --lamb_conv_weight_L2 1e-7 --lamb_bn_weight_L1 1e-7 --lamb_bn_weight_L2 1e-7 --report-on-test-set',
    'python cath.py --data-filename cath_10arch_ca_reducedx2.npz --model ResNet34 --training-epochs 200 --batch-size 16 --batchsize-multiplier 1 --randomize-orientation --kernel-size 3 --initial_lr=0.0005 --lr_decay_start=80 --burnin-epochs 80 --lr_decay_base=.97 --downsample-by-pooling --p-drop-conv 0.01125 --report-frequency 2 --lamb_conv_weight_L1 3.16e-7 --lamb_conv_weight_L2 3.16e-7 --lamb_bn_weight_L1 3.16e-7 --lamb_bn_weight_L2 3.16e-7 --report-on-test-set',
    'python cath.py --data-filename cath_10arch_ca_reducedx4.npz --model ResNet34 --training-epochs 400 --batch-size 16 --batchsize-multiplier 1 --randomize-orientation --kernel-size 3 --initial_lr=0.0005 --lr_decay_start=160 --burnin-epochs 160 --lr_decay_base=.985 --downsample-by-pooling --p-drop-conv 0.01250 --report-frequency 4 --lamb_conv_weight_L1 1.0e-6 --lamb_conv_weight_L2 1.0e-6 --lamb_bn_weight_L1 1.0e-6 --lamb_bn_weight_L2 1.0e-6 --report-on-test-set',
    'python cath.py --data-filename cath_10arch_ca_reducedx8.npz --model ResNet34 --training-epochs 800 --batch-size 16 --batchsize-multiplier 1 --randomize-orientation --kernel-size 3 --initial_lr=0.0005 --lr_decay_start=320 --burnin-epochs 320 --lr_decay_base=.992 --downsample-by-pooling --p-drop-conv 0.01375 --report-frequency 8 --lamb_conv_weight_L1 3.16e-6 --lamb_conv_weight_L2 3.16e-6 --lamb_bn_weight_L1 3.16e-6 --lamb_bn_weight_L2 3.16e-6 --report-on-test-set',
    'python cath.py --data-filename cath_10arch_ca_reducedx16.npz --model ResNet34 --training-epochs 1600 --batch-size 16 --batchsize-multiplier 1 --randomize-orientation --kernel-size 3 --initial_lr=0.0005 --lr_decay_start=640 --burnin-epochs 640 --lr_decay_base=.996 --downsample-by-pooling --p-drop-conv 0.015 --report-frequency 16 --lamb_conv_weight_L1 1.0e-5 --lamb_conv_weight_L2 1.0e-5 --lamb_bn_weight_L1 1.0e-5 --lamb_bn_weight_L2 1.0e-5 --report-on-test-set',
    'python cath.py --data-filename cath_10arch_ca.npz --model SE3ResNet34Small --training-epochs 100 --batch-size 16 --batchsize-multiplier 1 --randomize-orientation --kernel-size 3 --initial_lr=0.1 --lr_decay_start=40 --burnin-epochs 40 --lr_decay_base=.94 --downsample-by-pooling --p-drop-conv 0.1 --report-frequency 1 --lamb_conv_weight_L1 1e-8 --lamb_conv_weight_L2 1e-8 --lamb_bn_weight_L1 1e-8 --lamb_bn_weight_L2 1e-8 --report-on-test-set',
    'python cath.py --data-filename cath_10arch_ca_reducedx2.npz --model SE3ResNet34Small --training-epochs 200 --batch-size 16 --batchsize-multiplier 1 --randomize-orientation --kernel-size 3 --initial_lr=0.1 --lr_decay_start=80 --burnin-epochs 80 --lr_decay_base=.97 --downsample-by-pooling --p-drop-conv 0.1 --report-frequency 2 --lamb_conv_weight_L1 1.78e-8 --lamb_conv_weight_L2 1.78e-8 --lamb_bn_weight_L1 1.78e-8 --lamb_bn_weight_L2 1.78e-8 --report-on-test-set',
    'python cath.py --data-filename cath_10arch_ca_reducedx4.npz --model SE3ResNet34Small --training-epochs 400 --batch-size 16 --batchsize-multiplier 1 --randomize-orientation --kernel-size 3 --initial_lr=0.1 --lr_decay_start=160 --burnin-epochs 160 --lr_decay_base=.985 --downsample-by-pooling --p-drop-conv 0.1 --report-frequency 4 --lamb_conv_weight_L1 3.16e-8 --lamb_conv_weight_L2 3.16e-8 --lamb_bn_weight_L1 3.16e-8 --lamb_bn_weight_L2 3.16e-8 --report-on-test-set',
    'python cath.py --data-filename cath_10arch_ca_reducedx8.npz --model SE3ResNet34Small --training-epochs 800 --batch-size 16 --batchsize-multiplier 1 --randomize-orientation --kernel-size 3 --initial_lr=0.1 --lr_decay_start=320 --burnin-epochs 320 --lr_decay_base=.992 --downsample-by-pooling --p-drop-conv 0.1 --report-frequency 8 --lamb_conv_weight_L1 5.62e-8 --lamb_conv_weight_L2 5.62e-8 --lamb_bn_weight_L1 5.62e-8 --lamb_bn_weight_L2 5.62e-8 --report-on-test-set',
    'python cath.py --data-filename cath_10arch_ca_reducedx16.npz --model SE3ResNet34Small --training-epochs 1600 --batch-size 16 --batchsize-multiplier 1 --randomize-orientation --kernel-size 3 --initial_lr=0.1 --lr_decay_start=640 --burnin-epochs 640 --lr_decay_base=.996 --downsample-by-pooling --p-drop-conv 0.1 --report-frequency 16 --lamb_conv_weight_L1 1e-7 --lamb_conv_weight_L2 1e-7 --lamb_bn_weight_L1 1e-7 --lamb_bn_weight_L2 1e-7 --report-on-test-set'
    ]

for experiment_call in experiment_calls:
    os.system(experiment_call)