from utils import *

# raw_filename = 'SimulatedERP_raw.fif'
# try:
# 	raw = mne.io.read_raw_fif(raw_filename, preload=True)
# 	event_id = {'CondZero':1, 'CondOne':2}
# except FileNotFoundError:
# 	#if not make the raw simulation and save
# 	raw,event_id = SimulateRaw(amp1=10, amp2=5, freq=2., batch=1)
# 	raw.save(raw_filename,overwrite=True)

raw,event_id = SimulateRaw(amp1=100, amp2=50, freq=2., batch=2)
epochs = PreProcess(raw,event_id,filter_data=False,plot_erp=True)
feats = FeatureEngineer(epochs,model_type='NN')
model,_ = CreateModel(feats, units=[8,8,8])
TrainTestVal(model,feats)

