from utils import *






data_dir = '/Users/kylemathewson/Desktop/data/'
exp = 'P3'
subs = [ '001']
sessions = ['ActiveWet']
nsesh = len(sessions)
event_id = {'Target': 1, 'Standard': 2}

sub = subs[0]
session = sessions[0]
raw = LoadBVData(sub,session,data_dir,exp)


epochs = PreProcess(raw,event_id,
				emcp_epochs=False, rereference=True,
				plot_erp=True, rej_thresh_uV=250, 
				epoch_time=(-.2,1), baseline=(-.2,0) )
epochs_new = PreProcess(raw,event_id,
				emcp_epochs=True, rereference=True,
				plot_erp=True, rej_thresh_uV=250, 
				epoch_time=(-.2,1), baseline=(-.2,0) )


#plot results
epochs['Target'].plot()
epochs_new['Target'].plot()



