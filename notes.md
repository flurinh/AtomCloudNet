https://stackoverflow.com/questions/57062757/graph-from-smiles --> create graphs from molecules

nx.adjacency_matrix(mol, weight='order').todense()


https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#


How to setup leonhard:

leo: module load eth_proxy

local: ssh-keygen

leo: mkdir $HOME/.ssh
leo: chmod 700 $HOME/.ssh

local: /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
local: brew install lftp

copy files to leon: 
lftp sftp://hidberf@login.leonhard.ethz.ch -e "mirror -v -R -P 8 /users/modlab/PycharmProjects/Protos/data /cluster/home/hidberf/Protos/data; exit"

cat /Users/modlab/PycharmProjects/.keys.pub | ssh hidberf@login.leonhard.ethz.ch "cat - >> .ssh/authorized_keys"


module load cuda/10.0.130 cudnn/7.5
module load python_gpu/3.7.1
module load gcc/6.3.0
--> I might have to load this every time i log in!


https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html --> go through the steps


Ideas: 

Voxel network:      https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.voxel_grid
Point graph:        ...
Points and fields:  Try to find representation of forcefields 
