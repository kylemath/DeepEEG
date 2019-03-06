from utils import *

# raw_filename = 'SimulatedERP_raw.fif'
# try:
# 	raw = mne.io.read_raw_fif(raw_filename, preload=True)
# 	event_id = {'CondZero':1, 'CondOne':2}
# except FileNotFoundError:
# 	#if not make the raw simulation and save
# 	raw,event_id = SimulateRaw(amp1=10, amp2=5, freq=2., batch=1)
# 	raw.save(raw_filename,overwrite=True)

raw,event_id = SimulateRaw(amp1=100, amp2=50, freq=2., batch=1)
epochs = PreProcess(raw,event_id,filter_data=False,plot_erp=False,
					epoch_time=(-1,2))

pick = 35
for event in event_id.keys():
	fig = plt.imshow(epochs[event]._data[:,pick,:])
	plt.show()

feats = FeatureEngineer(epochs,model_type='NN',
						frequency_domain=True,include_phase=True)
model,_ = CreateModel(feats, units=[64,32,16])
TrainTestVal(model,feats)

