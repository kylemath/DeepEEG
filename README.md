## DeepEEG ##

![DeepEEG Image](DeepEEGImage2.png)
 
Keras/Tensorflow deep learning stacks that processes EEG trials or raw files from the MNE toolbox as input and predicts binary trial category as output (could scale to multiclass?). This is all made to run on Google Colab notebooks using cloud GPU capabilities, so the git repo's get loaded at the start of the code into the workspace. Minor mods may be needed to use local Jupyter notebook. Long term goal of command line interface and mne toolbox plugin.

Strategy:
* Load in Brain Products or Interaxon Muse files with mne as mne.raw, 
* PreProcess(mne.raw) - normal ERP preprocessing to get trials by time by electrode mne.epochs
* FeatureEngineer(mne.epochs) - Either time domain or frequency domain feature extraction in DeepEEG.Feats class
* CreateModel(DeepEEG.Feats) - Customizes DeepEEG.Model for input data, pick from NN, CNN, LSTM, or AutoEncoders, splits data
* TrainTestVal(DeepEEG.Feats,DeepEEG.Model) - Train the model, validate it during training, and test it once complete, Plot loss during learning and at test
                                             
Dataset example:
* Interaxon Muse - eeg-notebooks -  https://github.com/kylemath/eeg-notebooks
* Brain Recorder Data 

API:
* Input the data directory and subject numbers of any eeg-notebook experiment (https://github.com/kylemath/eeg-notebooks)
* Load in .vhdr brain products files by filename with mne io features
* FeatureEngineer can load any mne Epoch object too - https://martinos.org/mne/stable/generated/mne.Epochs.html

LearningModels:
* First try basic Neural Network (NN)
* Then try Convolution Neural Net (CNN)
* Then try Long-Short Term Memory Recurrant Neural Net (LSTM)
* Can also try using (AUTO) or (AUTODeep) to clean eeg data, or create features for other models

DataModels:
* Try subject specific models 
* Then pool data over all subjects
* Then try multilevel models (in the works)

Using: 
* https://github.com/kylemath/eeg-notebooks
* https://github.com/mne-tools/mne-python
* https://github.com/keras-team/keras/blob/master/examples/imdb_cnn_lstm.py
* https://github.com/ml4a/ml4a-guides/blob/master/notebooks/keras_classification.ipynb
* https://github.com/tevisgehr/EEG-Classification

Resources:
* https://arxiv.org/pdf/1901.05498.pdf 
* http://proceedings.mlr.press/v56/Thodoroff16.pdf
* https://arxiv.org/abs/1511.06448
* https://github.com/ml4a
* http://oxfordre.com/neuroscience/view/10.1093/acrefore/9780190264086.001.0001/acrefore-9780190264086-e-46
