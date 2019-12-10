Synful
======
Overview
--------
Synful: A project for the automated detection of synaptic partners in Electron Microscopy brain data using U-Nets (type of Convolutional Neural Network).

This repos provides train and predict scripts for synaptic partner detection. For more details, see the underlying publication:

- [Automatic Detection of Synaptic Partners in a Whole-Brain Drosophila EM Dataset](biorxiv link)


We used the method to predict 244 Million synaptic partners in the full adult fly brain (FAFB) dataset.
Please see https://github.com/funkelab/synfulfafb for data dissemination, and benchmark datasets.

Note, that this repos is work-in-progress. Please don't hesitate to file an issue or write us an email if something is unclear.

- [ ] Add train scripts
- [ ] Add inference scripts
- [ ] Add download links for pretrained models

Method
------
The pipeline processes 3D raw data in two steps into synaptic partners:
  1) inference of a) `syn_indicator_mask` (postsynaptic locations) and b) `direction_vector` (vector pointing from postsynaptic location to its presynaptic partner)
  2) synapse extraction: a) locations extractions based on `syn_indicator_mask` and b) finding presynaptic partner based on `direction_vector`

For CREMI-like (FAFB) data (anisotropic EM data from the Fruit Fly), pretrained models are available. For all other data, models need to be trained from scratch.

![method_figure](docs/_static/method_overview.png)

Training
--------

Training scripts are found in

```
examples/training/<setup>/
```

where `<setup>` is the name of a particular network configuration.
In such a <setup> directory, you will find three files:
- `train.py` (call to start training)
- `generate_network.py` (is called by train.py and generates tensorflow train network)
- `predict.py` (script used for inference)

To get started, have a look at the train script in [examples/training/train.py](02_train/base).


Inference
--------
Will be added soon.