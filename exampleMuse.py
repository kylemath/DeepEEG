from utils import *
load_verbose = True
data_dir = 'visual/cueing'
event_names = ['LeftCue','RightCue']
subs = [101,102]
nsesh = 2
sfreq = 256.
event_nums = [1,2]


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





#nsubs = len(subs)
#raw = []
#print('Loading Data')
#	print('Subject number ' + str(isub+1) + '/' + str(nsubs))
#	for isesh in range(nsesh):
#		print(' Session number ' + str(isesh+1) + '/' + str(nsesh))
#		if isub == 0 and isesh == 0:
#			raw.append(muse_load_data(data_dir,sfreq=sfreq,
#						subject_nb=isub,session_nb=nsesh,
#						verbose=load_verbose))

