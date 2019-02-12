from utils import *
import mne
from mne import read_source_spaces, find_events, Epochs, compute_covariance
from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_raw

print(__doc__)

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
src_fname = data_path + '/subjects/sample/bem/sample-oct-6-src.fif'
bem_fname = (data_path + 
			'/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif')

raw = mne.io.read_raw_fif(raw_fname)
raw.set_eeg_reference(projection=True)
raw = raw.crop(0., 255.)
	
n_dipoles = 1
epoch_duration = 2.
n = 0
sfreq = raw.info['sfreq']
ampzero = 25
ampone = 50

def data_fun_zero(times):
	n = 0
	n_samp = len(times)
	window = np.zeros(n_samp)
	start, stop = [int(ii * float(n_samp) / (2 * n_dipoles))
					for ii in (2 * n, 2 * n + 1)]
	window[start:stop] = np.hamming(stop-start)
	n = 1
	data = ampzero * 1e-9 * np.sin(2. * np.pi * 1. * n * times)
	data *= window 
	return data

def data_fun_one(times):
	n = 0
	n_samp = len(times)
	window = np.zeros(n_samp)
	start, stop = [int(ii * float(n_samp) / (2 * n_dipoles))
					for ii in (2 * n, 2 * n + 1)]
	window[start:stop] = np.hamming(stop-start)
	n = 1
	data = ampone * 1e-9 * np.sin(2. * np.pi * 1. * n * times)
	data *= window 
	return data

times = raw.times[:int(sfreq * epoch_duration)]
src = read_source_spaces(src_fname)
stc_zero = simulate_sparse_stc(src, n_dipoles=n_dipoles, times=times,
						data_fun=data_fun_zero, random_state=0)
stc_one = simulate_sparse_stc(src, n_dipoles=n_dipoles, times=times,
						data_fun=data_fun_one, random_state=0)


raw_sim_zero = simulate_raw(raw, stc_zero, trans_fname, src, bem_fname, 
					cov='simple', iir_filter=[0.2, -0.2, 0.04],
					ecg=True, blink=True, n_jobs=1, verbose=True)
raw_sim_one = simulate_raw(raw, stc_one, trans_fname, src, bem_fname, 
					cov='simple', iir_filter=[0.2, -0.2, 0.04],
					ecg=True, blink=True, n_jobs=1, verbose=True)

event_id = {'CondZero': 1,'CondOne': 2}
#replace 1 with 2 in second dataset
stim_pick = raw_sim_one.info['ch_names'].index('STI 014')
raw_sim_one._data[stim_pick][np.where(raw_sim_one._data[stim_pick]==1)] = 2
raw = concatenate_raws([raw_sim_zero, raw_sim_one])



epochs = PreProcess(raw,event_id,
		plot_events=False,
		epoch_time=(-0.2,2))


#events = find_events(raw)

#picks = mne.pick_types(raw.info, eeg=True, eog=True, meg=False, exclude='bads')
#epochs = Epochs(raw, events, event_id, -0.2, epoch_duration, 
#				picks=picks, preload=True)
#evoked = epochs.average()

#plot individual conditions evoked here

feats = FeatureEngineer(epochs)
model,_ = CreateModel(feats, units=[16,16])
TrainTestVal(model,feats)

