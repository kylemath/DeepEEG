from utils import *
raw,event_id = SimulateRaw(amp1=50, amp2=60, freq=1.)
epochs = PreProcess(raw,event_id)
feats = FeatureEngineer(epochs)
model,_ = CreateModel(feats, units=[16,16])
TrainTestVal(model,feats)

