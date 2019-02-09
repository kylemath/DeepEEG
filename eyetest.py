from utils import *
data_dir = 'visual/cueing'
subs = [101]
nsesh = 1
event_id = {'LeftCue': 1,'RightCue': 2}
raw = LoadMuseData(subs,nsesh,data_dir)
raw = GrattonEmcp(raw)