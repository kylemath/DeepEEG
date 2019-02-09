#conda create -n deepeeg
#source activate deepeeg
#chomd +x install.sh
#bash install.sh
#!git clone https://github.com/kylemath/eeg-notebooks
#python
from utils import *
data_dir = '/Users/kylemathewson/Desktop/'
exp = 'bikepark'
subs = [ '009']
sessions = ['quiet','traffic']
nsesh = len(sessions)
event_id = {'Standard': 1, 'Target': 2}

#Load Data
raw = LoadBVData(subs,sessions,data_dir,exp)
#Pre-Process EEG Data
epochs = PreProcess(raw,event_id,emcp=True,rereference=True,
					epoch_decim=10)
#Engineer Features for Model
feats = FeatureEngineer(epochs,model_type='LSTM',electrode_median=True)
#Create Model
model,_ = CreateModel(feats, units=[8,8], dropout=.05)
#Train with validation, then Test
TrainTestVal(model,feats)


