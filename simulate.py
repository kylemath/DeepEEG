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
def data_fun(times):
	global n 
	n_samp = len(times)
	window = np.zeros(n_samp)
	start, stop = [int(ii * float(n_samp) / (2 * n_dipoles))
					for ii in (2 * n, 2 * n + 1)]
	window[start:stop] = 1.
	n += 1
	data = 25e-9 * np.sin(2. * np.pi * 10. * n * times)
	data *= window 
	return data

times = raw.times[:int(sfreq * epoch_duration)]
src = read_source_spaces(src_fname)
stc = simulate_sparse_stc(src, n_dipoles=n_dipoles, times=times,
						data_fun=data_fun, random_state=0)

raw_sim = simulate_raw(raw, stc, trans_fname, src, bem_fname, 
					cov='simple', iir_filter=[0.2, -0.2, 0.04],
					ecg=True, blink=True, n_jobs=1, verbose=True)

events = find_events(raw_sim)
picks = mne.pick_types(raw.info, eeg=True, eog=True, meg=False, exclude='bads')
epochs = Epochs(raw_sim, events, 1, -0.2, epoch_duration, 
				picks=picks, preload=True)
evoked = epochs.average()




fig = plt.figure()
ax1 = plt.subplot(2,1,1)
ax1.plot(times, 1e9 * stc.data.T)
ax1.set(ylabel='Amplitude (nAm)', xlabel='Time (sec)')

ax2 = plt.subplot(2,1,2)
ax2 = plt.plot(evoked.times,evoked._data.T)
mne.viz.utils.plt_show()
