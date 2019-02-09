#conda create -n deepeeg
#source activate deepeeg
#chomd +x install.sh
#bash install.sh
#!git clone https://github.com/kylemath/eeg-notebooks
#python
from utils import *
data_dir = '/Users/kylemathewson/Desktop/'
exp = 'bikepark'
subs = [ '005']
sessions = ['quiet','traffic']
nsesh = len(sessions)
event_id = {'Standard': 1, 'Target': 2}

#Load Data
raw = LoadBVData(subs,sessions,data_dir,exp)
#Pre-Process EEG Data
epochs = PreProcess(raw,event_id,emcp=True)
#Engineer Features for Model
feats = FeatureEngineer(epochs)
#Create Model
model,_ = CreateModel(feats)
#Train with validation, then Test
TrainTestVal(model,feats)


