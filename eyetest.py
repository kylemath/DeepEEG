from utils import *
data_dir = '/Users/kylemathewson/Desktop/'
exp = 'bikepark'
subs = [ '009']
sessions = ['quiet','traffic']
nsesh = len(sessions)
event_id = {'Standard': 1, 'Target': 2}
raw = LoadBVData(subs,sessions,data_dir,exp)
raw.plot(scalings='auto')
raw = GrattonEmcpRaw(raw)
raw.plot(scalings='auto')
