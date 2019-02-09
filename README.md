## DeepEEG ##

MNE/Keras/Tensorflow library for classification of EEG data

* [Kyle E. Mathewson](https://github.com/kylemath)
* [Kory W. Mathewson](https://github.com/korymath)

![DeepEEG Image](DeepEEGImage2.png)

DeepEEG is a Keras/Tensorflow deep learning library that processes EEG trials or raw files from the MNE toolbox as input and predicts binary trial category as output (could scale to multiclass?). This is all made to run on Google Colab notebooks using cloud GPU capabilities, so the git repo's get loaded at the start of the code into the workspace. Minor modifications may be needed to use local Jupyter notebook. Long term goal of command line interface and MNE toolbox plugin.

Colab Notebook Example:
https://colab.research.google.com/github/kylemath/DeepEEG/blob/master/notebooks/DeepEEG.ipynb

## Getting Started Locally:

DeepEEG is tested on macOS 10.14 with Python3.
Prepare your environment the first time:

```sh
# using conda
conda create -n deepeeg
source activate deepeeg
# using virtualenv
# python3 -m venv deepeeg
# source deepeeg/bin/activate
```

```sh
git clone https://github.com/kylemath/DeepEEG/
cd DeepEEG
./install.sh
git clone https://github.com/kylemath/eeg-notebooks
```

You are now ready to run DeepEEG.

This loads in some example data from eeg-notebooks

```python
from utils import *
data_dir = 'visual/cueing'
subs = [101,102]
nsesh = 2
event_id = {'LeftCue': 1,'RightCue': 2}
```
Load muse data, preprocess into trials,prepare for model, create model, and train and test model

```python
#Load Data
raw = LoadMuseData(subs,nsesh,data_dir)
```

```python
#Pre-Process EEG Data
epochs = PreProcess(raw,event_id)
```

```python
#Engineer Features for Model
feats = FeatureEngineer(epochs)
```

```python
#Create Model
model,_ = CreateModel(feats)
```

```python
#Train with validation, then Test
TrainTestVal(model,feats)
```

## Tests

You can run the unittests with the following command:
```
python -m unittest
```

## Strategy
* Load in Brain Products or Interaxon Muse files with mne as mne.raw,
* PreProcess(mne.raw) - normal ERP preprocessing to get trials by time by electrode mne.epochs
* FeatureEngineer(mne.epochs) - Either time domain or frequency domain feature extraction in DeepEEG.Feats class
* CreateModel(DeepEEG.Feats) - Customizes DeepEEG.Model for input data, pick from NN, CNN, LSTM, or AutoEncoders, splits data
* TrainTestVal(DeepEEG.Feats,DeepEEG.Model) - Train the model, validate it during training, and test it once complete, Plot loss during learning and at test

## Dataset example
* Interaxon Muse - eeg-notebooks -  https://github.com/kylemath/eeg-notebooks
* Brain Recorder Data

## API
* Input the data directory and subject numbers of any eeg-notebook experiment (https://github.com/kylemath/eeg-notebooks)
* Load in .vhdr brain products files by filename with mne io features
* FeatureEngineer can load any mne Epoch object too - https://martinos.org/mne/stable/generated/mne.Epochs.html

## Preprocessing
* To be moved to another repo eventually
* Bandpass filter
* Regression Eye movement correction (if eye channels)
  - EOG Regression example: https://cbrnr.github.io/2017/10/20/removing-eog-regression/
  - Original emcp paper: https://apps.dtic.mil/dtic/tr/fulltext/u2/a125699.pdf
* Epoch segmentation (time limits, baseline correction)
* Artifact rejection

## LearningModels
* First try basic Neural Network (NN)
* Then try Convolution Neural Net (CNN)
* New is a 3D convolutional NN (CNN3D) in the frequency domain
* Then try Long-Short Term Memory Recurrant Neural Net (LSTM)
* Can also try using (AUTO) or (AUTODeep) to clean eeg data, or create features for other models

## DataModels
* Try subject specific models
* Then pool data over all subjects
* Then try multilevel models (in the works)

## Code References
* https://github.com/kylemath/eeg-notebooks
* https://github.com/mne-tools/mne-python
* https://github.com/keras-team/keras/blob/master/examples/imdb_cnn_lstm.py
* https://github.com/ml4a/ml4a-guides/blob/master/notebooks/keras_classification.ipynb
* https://github.com/tevisgehr/EEG-Classification

## Resources
* https://arxiv.org/pdf/1901.05498.pdf
* http://proceedings.mlr.press/v56/Thodoroff16.pdf
* https://arxiv.org/abs/1511.06448
* https://github.com/ml4a
* http://oxfordre.com/neuroscience/view/10.1093/acrefore/9780190264086.001.0001/acrefore-9780190264086-e-46
* https://arxiv.org/pdf/1811.10111.pdf
