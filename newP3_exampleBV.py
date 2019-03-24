from utils import *
data_dir = '/Users/kylemathewson/data/'
exp = 'P3'
subs = ['001','002','004','005','006','007','008','010']
sessions = ['ActiveDry','ActiveWet','PassiveWet']

event_id = {'Target': 1, 'Standard': 2}
epochs = {} #dict
all_evokeds = {}
fig = plt.figure()

for i_session, session in enumerate(sessions):
	all_lists = {'Target':[], 'Standard':[]}
	for sub in subs:
		print('Loading data for subject ' + sub)
		#Load Data
		raw = LoadBVData(sub,session,data_dir,exp)
		#Pre-Process EEG Data
		epochs[session] = PreProcess(raw,event_id,
							emcp_epochs=True, rereference=True,
							plot_erp=False, rej_thresh_uV=500, 
							epoch_time=(-.2,1), baseline=(-.2,0) )
		#create evoked dict by averaging all the epochs for this test subject
		all_lists['Target'].append(epochs[session]['Target'].average())
		all_lists['Standard'].append(epochs[session]['Standard'].average())
	#create the dict with condition keys and values are lists over subs	
	all_evokeds['Target'] = all_lists['Target']
	all_evokeds['Standard'] = all_lists['Standard']
	ax = plt.subplot(3,1,i_session+1)
	viz.plot_compare_evokeds(all_evokeds,picks=[6],axes=ax,
							title=session,show=False)
plt.show()




