#conda create -n deepeeg
#source activate deepeeg
#bash install.sh
#!git clone https://github.com/kylemath/eeg-notebooks
from utils import *
load_verbose = True
data_dir = 'visual/cueing'
subs = [101,102]
nsesh = 2
sfreq = 256.
event_names = ['LeftCue','RightCue']
event_nums = [1,2]
event_id = {event_names[1]: event_nums[1],
			event_names[0]: event_nums[0]}

#Load Data
nsubs = len(subs)
raw = []
print('Loading Data')
for isub,sub in enumerate(subs):
	print('Subject number ' + str(isub+1) + '/' + str(nsubs))
	for isesh in range(nsesh):
		print(' Session number ' + str(isesh+1) + '/' + str(nsesh))
		raw.append(muse_load_data(data_dir, sfreq=sfreq ,subject_nb=sub, 
								  session_nb=isesh+1,verbose=load_verbose))


#Preprocess
raw = concatenate_raws(raw)
epochs = PreProcess(raw,event_id)

