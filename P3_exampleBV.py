#conda create -n deepeeg
#source activate deepeeg
#chomd +x install.sh
#bash install.sh
#!git clone https://github.com/kylemath/eeg-notebooks
#python
from utils import *
data_dir = '/Users/kylemathewson/Desktop/data/'
exp = 'P3'
subs = ['001','002','004','005','006','007','008','010']
#subs = [ '002']

sessions = ['ActiveDry','ActiveWet','PassiveWet']

nsesh = len(sessions)
event_id = {'Target': 1, 'Standard': 2}

epochs = []
for sub in subs:
	print('Loading data for subject ' + sub)
	for session in sessions:
		#Load Data
		raw = LoadBVData(sub,session,data_dir,exp)
		#Pre-Process EEG Data
		temp_epochs = PreProcess(raw,event_id,
							emcp_epochs=True, rereference=True,
							plot_erp=False, rej_thresh_uV=1000, 
							epoch_time=(-.2,1), baseline=(-.2,0), 
							epoch_decim=10 )
		if len(temp_epochs) > 0:
			epochs.append(temp_epochs)
		else:
			print('Sub ' + sub + ', Cond ' 
					+ session + 'all trials rejected')

epochs = concatenate_epochs(epochs)	

#Engineer Features for Model
feats = FeatureEngineer(epochs,model_type='NN',electrode_median=False,
 						frequency_domain=False)
#Create Model
model,_ = CreateModel(feats, units=[64,32,16,8])
#Train with validation, then Test
TrainTestVal(model,feats)


