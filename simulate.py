from utils import *

raw_filename = 'SimulatedERP_raw.fif'

try:
	raw = mne.io.read_raw_fif(raw_filename, preload=True)
	event_id = {'CondZero':1, 'CondOne':2}
except FileNotFoundError:
	#if not make the raw simulation and save
	raw,event_id = SimulateRaw(amp1=50, amp2=60, freq=1.)
	raw.save(raw_filename,overwrite=True)

epochs = PreProcess(raw,event_id,filter_data=False)
feats = FeatureEngineer(epochs)
model,_ = CreateModel(feats, units=[16,16])
TrainTestVal(model,feats)

