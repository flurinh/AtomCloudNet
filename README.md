# AtomCloudNet

Our network architecture is looking at the neighborhood of each atom in a given chemical space. The surrounding "cloud" of atoms then feed into the generation of new features describing the whole state of the atom.
After multiple steps of updating atom-features based on surrounding clouds features propagate through the molecule reaching more distant atoms.

The resulting (depending on the network architecture intermediate) molecular representation could be used to investigate specific features of atoms or they
can be collapsed and fed into a feed forward architecture to derive chemical properties of the whole molecule.

The "AtomCloudNet"-architecture is based on the point-convolution architecture which is highly performant in similar tasks in machine learning, such as
 surface segmentation or object detection.

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
features (as shown with similar approaches, e.g. Schnet, Schnorb) in a fraction of the time required to calculate them.
This could open the path to real-time calculation of highly informative descriptions of chemical space and give rise 
to more performant models by using these resultant approximate features for the prediction of chemical properties.

## Challenge of AtomNet

There are several challenges in making this architecture work. Choosing the neighborhood of an atom for example is not done via
calculation of euclidean distance, but instead selection of the coulomb-environment (closest neighbor has highest coulomb-
interaction). 

The most difficult task is "How to combine the neighborhoods atoms features in a meaningful, descriptive way?" 
AtomCloudNet takes neighbors features, stacks them and applies a convolution layer (similar to PointNet++).
However given the high relevance of spatial directions in modelling relations in AtomClouds we consider the implementation of
kernels in 3D space to provide easy access to modelling the chemical space.
