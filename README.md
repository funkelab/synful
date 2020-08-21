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
- [x] Add inference scripts
- [x] Add download links for pretrained models

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

Once you trained a network, you can use this script to run inference:

```
cd scripts/predict/
python predict_blockwise.py predict_template.json
```
Adapt following parameters in the configfile <scripts/predict/predict_template.json>:
- `db_host` --> Put here the name of your running mongodb server (this is used to track which chunks are processed)
- `raw_file` --> Put here the filepath of your raw data (as an example you can use the CREMI data that you can download from www.cremi.org)

For a full list of parameters and explanation, see: <scripts/predict/predict_blockwise.py>.


#### Inference runtime

Processing a CREMI cube (5 microns X 5 microns x 5 microns) takes ~4 minutes on a single GPU.

Pretrained Models
-----------------
We provide pretrained models, that we discuss in detail in our [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2019.12.12.874172v2). You will find the results of our gridsearch and the parameters that we used in Figure 3 `Validation results on CREMI dataset`.

We provide four models that you can download from [here](https://www.dropbox.com/s/301382766164ism/pretrained.zip?dl=0).

Please extract the zip file into <scripts/train/> of this repository, this will add for each model a setup directory with the necassary config files, tensorflow checkpoint and predict script.

For instance for `p_setup52` (marked orange in Figure 3, one of the best performing models), you will get all relevant files in <scripts/train/p_setup52>.
To run inference, you have to change the setup parameter in the predict config file to `p_setup52` and proceed according to [inference section](#Inference).


#### Details about the provided models

|setup|specs|f-score with seg| f-score without|
|---|---|---|---|
|p_setup52 (+p_setup10)|big, curriculum, CE, ST|0.76|0.74|
|p_setup51|big, curriculum, CE, MT_2|0.76|0.73|
|p_setup54 (+p_setup05)|small, curriculum, MSE, ST|0.76|0.7|
|p_setup45 (+p_setup05)|small, standard, MSE, MT2|0.73|0.68|
Note, that for the models that have an underlying ST architecture we also indicate the setup for the corresponding direction-vector-models (p_setup05+p_setup10).
If you want to use the model with highest accuracy, pick `p_setup52`; If you want to use a model that gives reasonnable results, but also has fast inference runtime, pick `p_setup54`.