Synful
======
Overview
--------
Synful: A project for the automated detection of synaptic partners in Electron Microscopy brain data using U-Nets (type of Convolutional Neural Network).

This repository provides train and predict scripts for synaptic partner detection. For more details, see our [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2019.12.12.874172v1).

We used the method to predict 244 Million synaptic partners in the full adult fly brain (FAFB) dataset.
Please see https://github.com/funkelab/synful_fafb for data dissemination and benchmark datasets.

Please don't hesitate to open
an issue or write us an email ([Julia
Buhmann](mailto:buhmannj@janelia.hhmi.org) or [Jan
Funke](mailto:funkej@janelia.hhmi.org)) if you have any questions!

- [x] Add train scripts
- [ ] Add inference scripts
- [ ] Add download links for pretrained models

Method
------
The pipeline processes 3D raw data in two steps into synaptic partners:
  1) inference of a) `syn_indicator_mask` (postsynaptic locations) and b) `direction_vector` (vector pointing from postsynaptic location to its presynaptic partner)
  2) synapse extraction: a) locations extractions based on `syn_indicator_mask` and b) finding presynaptic partner based on `direction_vector`


![method_figure](docs/_static/method_overview.png)


System Requirements
-------------------

- Hardware requirements
  - training and prediction requires at least one GPU with sufficient memory (12 GB)
  - For instance, we mostly used `GeForce GTX TITAN X 12 GB` for our project
- Software requirements
  - Software has been tested on Linux (Ubuntu 16.04)

Installation Guide
------------------
from source (creating a conda env is optional, but recommended).
- Clone this repository.
- In a terminal:

```bash
conda create -n <conda_env_name> python=3.6
source activate <conda_env_name>
cd synful
pip install -r requirements.txt
python setup.py install
```
If you are interested in using the package for training and prediction, additionally add tensorflow and funlib.learn.tensorflow to your conda env:

```bash
conda install tensorflow-gpu=1.14 cudatoolkit=10.0
pip install git+git://github.com/funkelab/funlib.learn.tensorflow@0712fee6b6c083c6bfc86e76f475b2e40b3c64f2

```

#### Install time
Installation should take around 5 mins (including 3 mins for the tensorflow installation).


Training
--------

Training scripts are found in

```
train/<setup>
```

where `<setup>` is the name of a particular network configuration.
In such a <setup> directory, you will find two files:
- `generate_network.py` (generates a tensorflow network based on the parameter.json file in the same directoy)
- `train.py` (starts training)


To get started, have a look at the train script in [train/setup01/train.py](train/setup01).

To start training:
```bash
python generate_network.py parameter.json
python train.py parameter.json
```

- setup01: parameter.json is set to train a network on post-synaptic sites (single-task network)
- setup02: parameter.json is set to train on direction vectors (single-task network)
- setup03: parameter.json is set to train on both post-synaptic sites and direction vectors (multi-task network)

#### Training runtime
Training takes between 3 and 10 days (depending on the size of the network), but you should see reasonable results within a day (after 90k iterations).



### Monitoring Training

To visualize snapshots that are produced during training use this [script](scripts/visualization/visualize_snapshot.py):

```
python -i visualize_snapshot.py 300001 setup01
```

in order to load iteration `300001` of training setup `setup01` (use -1 to indicate most recent snapshot)


Inference
--------

Inference scripts will be added soon.
