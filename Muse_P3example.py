from utils import *
data_dir = 'visual/cueing'
subs = [101, 102, 103, 104]

# subs = [101, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112,
#         202, 203, 204, 205, 207, 208, 209, 210, 211, 
#         301, 302, 303, 304, 305, 306, 307, 308, 309,
#         1101, 1102, 1103, 1104, 1105, 1106, 1108, 1109, 1110,
#         1202, 1203, 1205, 1206, 1209, 1210, 1211, 1215,
#         1301, 1302, 1313, 
#         1401, 1402, 1403, 1404, 1405,  1408, 1410, 1411, 1412, 1413, 1413, 1414, 1415, 1416]

nsesh = 2
event_id = {'LeftCue': 1,'RightCue': 2}
#Load Data
raw = LoadMuseData(subs,nsesh,data_dir)
#Pre-Process EEG Data
epochs = PreProcess(raw,event_id)
#Engineer Features for Model
feats = FeatureEngineer(epochs)
#Create Model
model,_ = CreateModel(feats)
#Train with validation, then Test
TrainTestVal(model,feats)


