# ATOMCLOUDNET

Herein, we introduce AtomCloudNet (ACN), a deep learning model to predict molecular and atomic properties using
information of individual atomic environments extracted by point cloud convolution. We exemplify the capability of our
model by learning ground state atomization energies Urt of molecules calculated by first principal methods. Information
of atomic position Ri and nuclear charge Zi are mapped to a molecular property via applying spheric harmonic functions
Y m (φ, Θ). This resulted in a MAE of 131 kcal/mol. Therfore, ACN represents a novel approach for the prediction molec- l
ular and atomic properties.

The kernel has been described and implemented for pytorch [here](https://github.com/mariogeiger/se3cnn) and a useful tutorial
on the application can be found [here](https://blondegeek.github.io/e3nn_tutorial/).


## Dataset

We use the well-known [QM9 dataset](http://quantum-machine.org/datasets) to predict molecular properties and train our network.

## Installation

Create a conda environment: 
```
conda env create -f environment.yml
```
This installs most dependencies, but to run our code you will need the dependencies handling SE(3) point convolution:
```
"pip install git+https://github.com/AMLab-Amsterdam/lie_learn"
"pip install git+https://github.com/se3cnn/se3cnn"
```


## Analysis

Unzip the .zip data and .zip model folders.
Paths should be:

```
data / QM9Train / dsgdb9nsd_000049.xyz ...
data / QM9Test / dsgdb9nsd_000074.xyz ...
data / pkl / data_10000.pkl ...
model / run_140 / model_14001.pkl ...
```
To train a model according to its configuration file (e.g. '/config_ACN/config_13003.ini') run:
```
python sandbox.py --run 13003
```
To load a model and evaluate it on the testset:
```
python sandbox.py --run 13003 --mode 1
```
The results can be visualized by running:
```
python results.py
```


## AtomCloudNet Paper

Our paper about AtomCloudNet, some chemical insights and other literature.
[embed]https://github.com/flurinh/AtomCloudNet/blob/master/paper/AtomCloudNet.pdf



