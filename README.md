# AtomCloudNet

Our network architecture is looking at the neighborhood of each atom in a given chemical space. The surrounding "cloud" of atoms then feed into the generation of new features describing the whole state of the atom.
After updating atom-features based on its surroundings features propagate through the molecule reaching more distant atoms. Optionally after the cloud-convolution we can extend the feature-space by concatenating translation- and rotation-invariant features (e.g. sum over all 2- and 3-body-interactions) and the previously generated atomic features.

The atomic feature space is collapsed via global pooling over all atoms for each feature.

The resulting molecular representation could be used to investigate specific features of atoms or they
can be collapsed and fed into a feed forward architecture to derive chemical properties of the whole molecule.


## Why AtomClouds?

This type of model learns for a set of atoms in chemical space [(x, y, z), (Z)] (position, atom-embedding) an underlying 
representation. Advantages compared to other structure based approaches are:
- compared to full or sparse convolution this approach does neither depend on voxelizing the positional data into discrete spaces 
(which either leads to loss of information, due to positional error, or high computational costs, as 3D convolution scales very 
poorly with increasing resolution).
- compared to edge-graph-based models no connectivity information is required. This allows the application of our models to chemical input regions,
where we have disconnected graphs interacting (e.g. in a protein between disconnected aminoacids or between
protein and ligands, ...).
- compared to feature-selection based models: Some of the very best descriptions of chemical space are very time-consuming 
to obtain (DFT, coupled-cluster calculation). Our model could be trained to predict an approximation of such 
features in a fraction of the time required to calculate them.
This opens the path to real-time calculation of highly informative descriptions of chemical space and give rise 
to more performant models by using these resultant approximate features for the prediction of chemical properties.

## Challenge of AtomCloudNet

The most difficult task is "How to combine the neighborhoods atoms features in a meaningful, descriptive way?" 
AtomCloudNet uses a translation- and rotation-invariant kernel based on spherical harmonics.

The kernel has been described and implemented for pytorch here: https://github.com/mariogeiger/se3cnn


## Dataset

We use the well-known QM9 dataset (http://quantum-machine.org/datasets) to predict molecular properties and evaluate our network.