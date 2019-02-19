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
epochs = []
for isesh, session in enumerate(sessions):
	event_id = {(session + '/Target'): 1, (session + '/Standard'): 2}
	for sub in subs:
		print('Loading data for subject ' + sub)
		#Load Data
		raw = LoadBVData(sub,session,data_dir,exp)
		#Pre-Process EEG Data
		temp_epochs = PreProcess(raw,event_id,
							emcp=True, rereference=True,
							plot_erp=False, rej_thresh_uV=250, 
							epoch_time=(-1,2), baseline=(-1,-.5) )
		if len(temp_epochs) > 0:
			epochs.append(temp_epochs)
		else:
			print('Sub ' + sub + ', Cond ' 
					+ session + 'all trials rejected')



print(epochs)
epochs = concatenate_epochs(epochs)	
print(epochs)

#Engineer Features for Model
feats = FeatureEngineer(epochs,model_type='NN',electrode_median=False,
						frequency_domain=True)
#Create Model
model,_ = CreateModel(feats, units=[16,16])
#Train with validation, then Test
TrainTestVal(model,feats)


