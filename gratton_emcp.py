# raws | epochs = gratton_emcp(raws | epochs, settings)
# EOG Regression example: https://cbrnr.github.io/2017/10/20/removing-eog-regression/
# Finding EOG blinks in mne: https://martinos.org/mne/stable/auto_examples/preprocessing/plot_find_eog_artifacts.html
# timing of blinks in mne: https://martinos.org/mne/stable/auto_examples/preprocessing/plot_eog_artifact_histogram.html
# original emcp paper: https://apps.dtic.mil/dtic/tr/fulltext/u2/a125699.pdf

def gratton_emcp(data,datatype='raw'):
	if datatype == 'raw':
		raw = data
		raw_eeg = raw[:16,:][0]
		raw_eog = raw[16:18,:][0]
		b = np.linalg.solve(raw_eog @ raw_eog.T, raw_eog @ raw_eeg.T)
		print(b.shape)
		eeg_corrected = (raw_eeg.T - raw_eog.T @ b).T
		raw_new = raw.copy()
		raw_new._data[:16,:] = eeg_corrected
		data = raw_new
	if datatype == 'epochs':
		data = []
	return data


