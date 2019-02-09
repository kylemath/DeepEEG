#conda create -n deepeeg
#source activate deepeeg
#chomd +x install.sh
#bash install.sh
#!git clone https://github.com/kylemath/eeg-notebooks
#python
from utils import *
data_dir = '/Users/kylemathewson/Desktop/'
exp = 'bikepark'
#subs = [ '009']
subs = ['005', '007', '009', '010', '012', '013', '014', '015', '016', '019']

sessions = ['quiet','traffic']
nsesh = len(sessions)
event_id = {'Standard': 1, 'Target': 2}

#https://martinos.org/mne/stable/auto_tutorials/plot_visualize_evoked.html
epochs = []
for sub in subs:
	print('Loading data for subject ' + sub)
	#Load Data
	raw = LoadBVData(sub,sessions,data_dir,exp)
	#Pre-Process EEG Data
	epochs.append(PreProcess(raw,event_id,emcp=True,rereference=True,
						plot_erp=True))

epochs = concatenate_epochs(epochs)	
print(epochs)

#Engineer Features for Model
feats = FeatureEngineer(epochs,model_type='NN',electrode_median=False)
#Create Model
model,_ = CreateModel(feats, units=[16,16], dropout=.15)
#Train with validation, then Test
TrainTestVal(model,feats)


