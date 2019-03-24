from utils import *
data_dir = '/Users/kylemathewson/data/'
exp = 'P3'
subs = ['001','002','004','005','006','007','008','010']
subs = [ '002']

sessions = ['ActiveDry','ActiveWet','PassiveWet']
epochs = {} #dict
for session in sessions:
	event_id = {'Target': 1, 'Standard': 2}
	for sub in subs:
		print('Loading data for subject ' + sub)
		#Load Data
		raw = LoadBVData(sub,session,data_dir,exp)
		#Pre-Process EEG Data
		epochs[session] = PreProcess(raw,event_id,
							emcp_epochs=True, rereference=True,
							plot_erp=False, rej_thresh_uV=250, 
							epoch_time=(-1,2), baseline=(-1,-.5) )
		if len(epochs[session]) == 0:
			print('Sub ' + sub + ', Cond ' 
					+ session + 'all trials rejected')

print(epochs)

#create evoked dict by averaging all the epochs for this test subject
evoked_dict = {}
for session in sessions: 
	evoked_dict[session + '/' + 'Target'] = epochs[session]['Target'].average()
	evoked_dict[session + '/' + 'Standard'] = epochs[session]['Standard'].average()

#plot all the erps on one plot
viz.plot_compare_evokeds(evoked_dict['Target'],picks=[6])



