from numpy.random import seed
seed(1017)
from tensorflow import set_random_seed
set_random_seed(1017)

import os
from glob import glob
from collections import OrderedDict

import mne
from mne.io import RawArray
from mne import read_evokeds, read_source_spaces, compute_covariance
from mne import channels, find_events, concatenate_raws
from mne import pick_types, viz, io, Epochs, create_info
from mne import pick_channels, concatenate_epochs
from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_raw
from mne.channels import read_montage
from mne.time_frequency import tfr_morlet

import numpy as np
from numpy import genfromtxt

import pandas as pd
pd.options.display.precision = 4
pd.options.display.max_columns = None

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,12)

import keras
from keras import regularizers
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Flatten, Conv2D, MaxPooling2D, LSTM
from keras.layers import BatchNormalization, Conv3D, MaxPooling3D

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split


class Feats:
  def __init__(self, num_classes=2, class_weights=[1,1], input_shape=[16,], 
               new_times=1, model_type='1', 
               x_train=1, y_train=1, x_test=1, y_test=1, x_val=1, y_val=1):
    self.num_classes = num_classes
    self.class_weights = class_weights
    self.input_shape = input_shape
    self.new_times = new_times
    self.model_type = model_type
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.x_val = x_val
    self.y_val = y_val

def LoadBVData(sub,session,data_dir,exp):
  #for isub,sub in enumerate(subs):       
  print('Loading data for subject number: ' + sub)
  fname = data_dir + exp + '/' + sub + '_' + exp + '_' + session + '.vhdr'
  raw,sfreq = loadBV(fname,plot_sensors=False,plot_raw=False,
          plot_raw_psd=False,stim_channel=True)
  return raw

def LoadMuseData(subs, nsesh, data_dir, load_verbose=False, sfreq=256.):
  nsubs = len(subs)
  raw = []
  print('Loading Data')
  for isub,sub in enumerate(subs):
    print('Subject number ' + str(isub+1) + '/' + str(nsubs))
    for isesh in range(nsesh):
      print(' Session number ' + str(isesh+1) + '/' + str(nsesh))
      raw.append(muse_load_data(data_dir, sfreq=sfreq ,subject_nb=sub,
                    session_nb=isesh+1,verbose=load_verbose))
  raw = concatenate_raws(raw)
  return raw

def loadBV(filename, plot_sensors=True, plot_raw=True,
  plot_raw_psd=True, stim_channel=False, ):
  """Load in recorder data files."""


  #load .vhdr files from brain vision recorder
  raw = io.read_raw_brainvision(filename,
                          montage='standard_1020',
                          eog=('HEOG', 'VEOG'),
                          preload=True,stim_channel=stim_channel)

  #set sampling rate
  sfreq = raw.info['sfreq']
  print('Sampling Rate = ' + str(sfreq))

  #load channel locations
  print('Loading Channel Locations')
  if plot_sensors:
    raw.plot_sensors(show_names='True')

  ##Plot raw data
  if plot_raw:
    raw.plot(n_channels=16, block=True)

   #plot raw psd
  if plot_raw_psd:
    raw.plot_psd(fmin=.1, fmax=100 )

  return raw, sfreq

#from eeg-notebooks
def load_muse_csv_as_raw(filename, sfreq=256., ch_ind=[0, 1, 2, 3],
                         stim_ind=5, replace_ch_names=None, verbose=1):
    """Load CSV files into a Raw object.

    Args:
        filename (str or list): path or paths to CSV files to load

    Keyword Args:
        subject_nb (int or str): subject number. If 'all', load all
            subjects.
        session_nb (int or str): session number. If 'all', load all
            sessions.
        sfreq (float): EEG sampling frequency
        ch_ind (list): indices of the EEG channels to keep
        stim_ind (int): index of the stim channel
        replace_ch_names (dict or None): dictionary containing a mapping to
            rename channels. Useful when an external electrode was used.

    Returns:
        (mne.io.array.array.RawArray): loaded EEG
    """

    n_channel = len(ch_ind)

    raw = []
    for fname in filename:
        # read the file
        data = pd.read_csv(fname, index_col=0)

        # name of each channels
        ch_names = list(data.columns)[0:n_channel] + ['Stim']

        if replace_ch_names is not None:
            ch_names = [c if c not in replace_ch_names.keys()
                        else replace_ch_names[c] for c in ch_names]

        # type of each channels
        ch_types = ['eeg'] * n_channel + ['stim']
        montage = read_montage('standard_1005')

        # get data and exclude Aux channel
        data = data.values[:, ch_ind + [stim_ind]].T

        # convert in Volts (from uVolts)
        data[:-1] *= 1e-6

        # create MNE object
        info = create_info(ch_names=ch_names, ch_types=ch_types,
                           sfreq=sfreq, montage=montage, verbose=verbose)
        raw.append(RawArray(data=data, info=info, verbose=verbose))

    # concatenate all raw objects
    raws = concatenate_raws(raw, verbose=verbose)

    return raws

#from eeg-notebooks load_data
def muse_load_data(data_dir, subject_nb=1, session_nb=1, sfreq=256.,
                   ch_ind=[0, 1, 2, 3], stim_ind=5, replace_ch_names=None,
                   verbose=1):
    """Load CSV files from the /data directory into a Raw object.

    Args:
        data_dir (str): directory inside /data that contains the
            CSV files to load, e.g., 'auditory/P300'

    Keyword Args:
        subject_nb (int or str): subject number. If 'all', load all
            subjects.
        session_nb (int or str): session number. If 'all', load all
            sessions.
        sfreq (float): EEG sampling frequency
        ch_ind (list): indices of the EEG channels to keep
        stim_ind (int): index of the stim channel
        replace_ch_names (dict or None): dictionary containing a mapping to
            rename channels. Useful when an external electrode was used.

    Returns:
        (mne.io.array.array.RawArray): loaded EEG
    """


    if subject_nb == 'all':
        subject_nb = '*'
    if session_nb == 'all':
        session_nb = '*'

    data_path = os.path.join(
            'eeg-notebooks/data', data_dir,
            'subject{}/session{}/*.csv'.format(subject_nb, session_nb))
    fnames = glob(data_path)

    return load_muse_csv_as_raw(fnames,
                                sfreq=sfreq,
                                ch_ind=ch_ind,
                                stim_ind=stim_ind,
                                replace_ch_names=replace_ch_names,
                                verbose=verbose)


def SimulateRaw(amp1 = 50, amp2 = 100, freq = 1., batch=1):

  """Create simulated raw data and events of two kinds
  
  Keyword Args:
      amp1 (float): amplitude of first condition effect
      amp2 (float): ampltiude of second condition effect, 
          null hypothesis amp1=amp2
      freq (float): Frequency of simulated signal 1. for ERP 10. for alpha
      batch (int): number of groups of 255 trials in each condition
  Returns: 
      raw: simulated EEG MNE raw object with two event types
      event_id: dict of the two events for input to PreProcess()
  """


  data_path = sample.data_path()
  raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
  trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
  src_fname = data_path + '/subjects/sample/bem/sample-oct-6-src.fif'
  bem_fname = (data_path + 
        '/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif')

  
  raw_single = mne.io.read_raw_fif(raw_fname,preload=True)
  raw_single.set_eeg_reference(projection=True)
  raw_single = raw_single.crop(0., 255.)
  raw_single = raw_single.copy().pick_types(meg=False, eeg=True, eog=True, stim=True)

  #concatenate 4 raws together to make 1000 trials
  raw = []
  for i in range(batch):
    raw.append(raw_single)
  raw = concatenate_raws(raw)

  epoch_duration = 1.
  
  def data_fun(amp, freq):
    """Create function to create fake signal"""
    def data_fun_inner(times):
      """Create fake signal with no noise"""
      n_samp = len(times)
      window = np.zeros(n_samp)
      start, stop = [int(ii * float(n_samp) / 2)
        for ii in (0, 1)]
      window[start:stop] = np.hamming(stop - start)
      data = amp * 1e-9 * np.sin(2. * np.pi * freq * times)
      data *= window
      return data
    return data_fun_inner

  times = raw.times[:int(raw.info['sfreq'] * epoch_duration)]
  src = read_source_spaces(src_fname)

  stc_zero = simulate_sparse_stc(src, n_dipoles=1, times=times,
              data_fun=data_fun(amp1,freq), random_state=0)
  stc_one = simulate_sparse_stc(src, n_dipoles=1, times=times,
              data_fun=data_fun(amp2,freq), random_state=0)

  raw_sim_zero = simulate_raw(raw, stc_zero, trans_fname, src, bem_fname, 
            cov='simple', blink=True, n_jobs=1, verbose=True)
  raw_sim_one = simulate_raw(raw, stc_one, trans_fname, src, bem_fname, 
            cov='simple', blink=True, n_jobs=1, verbose=True)

  stim_pick = raw_sim_one.info['ch_names'].index('STI 014')
  raw_sim_one._data[stim_pick][np.where(raw_sim_one._data[stim_pick]==1)] = 2
  raw = concatenate_raws([raw_sim_zero, raw_sim_one])
  event_id = {'CondZero': 1,'CondOne': 2}
  return raw, event_id


def mastoidReref(raw):
  ref_idx = pick_channels(raw.info['ch_names'],['M2'])
  eeg_idx = pick_types(raw.info,eeg=True)
  raw._data[eeg_idx,:] =  raw._data[eeg_idx,:]  -  raw._data[ref_idx,:] * .5 ;
  return raw

def GrattonEmcpRaw(raw):
  raw_eeg = raw.copy().pick_types(eeg=True)[:][0]
  raw_eog = raw.copy().pick_types(eog=True)[:][0]
  b = np.linalg.solve(np.dot(raw_eog,raw_eog.T), np.dot(raw_eog,raw_eeg.T))
  eeg_corrected = (raw_eeg.T - np.dot(raw_eog.T,b)).T
  raw_new = raw.copy()
  raw_new._data[pick_types(raw.info,eeg=True),:] = eeg_corrected
  return raw_new



def PreProcess(raw, event_id, plot_psd=False, filter_data=True,
               eeg_filter_highpass=1, plot_events=False, epoch_time=(-.2,1),
               baseline=(-.2,0), rej_thresh_uV=200, rereference=False, 
               emcp=False, epoch_decim=1, plot_electrodes=False,
               plot_erp=False):



  sfreq = raw.info['sfreq']
  #create new output freq for after epoch or wavelet decim
  nsfreq = sfreq/epoch_decim
  tmin=epoch_time[0]
  tmax=epoch_time[1]
  eeg_filter_lowpass = nsfreq/2.5  #lower to avoid aliasing from decim

  #pull event names in order of trigger number
  event_names = ['A_error','B_error']
  i = 0
  for key, value in sorted(event_id.items(), key=lambda x: (x[1], x[0])):
    event_names[i] = key
    i += 1

  #Filtering
  if rereference:
    print('Rerefering to average mastoid')
    raw = mastoidReref(raw)

  if filter_data:
    print('Filtering Data')
    raw.filter(eeg_filter_highpass,eeg_filter_lowpass,
               method='iir', verbose='WARNING' )

  if plot_psd:
    raw.plot_psd(fmin=eeg_filter_highpass, fmax=nsfreq/2 )

  #Eye Correction
  if emcp:
    print('Eye Movement Correction')
    raw = GrattonEmcpRaw(raw)

  #Epoching
  events = find_events(raw,shortest_event=1)
  color = {1: 'red', 2: 'black'}
  #artifact rejection
  rej_thresh = rej_thresh_uV*1e-6

  #plot event timing
  if plot_events:
    viz.plot_events(events, sfreq, raw.first_samp, color=color,
                        event_id=event_id)

  #Constructevents
  epochs = Epochs(raw, events=events, event_id=event_id,
                  tmin=tmin, tmax=tmax, baseline=baseline,
                  preload=True,reject={'eeg':rej_thresh},
                  verbose=False, decim=epoch_decim)
  print('Remaining Trials: ' + str(len(epochs)))

  evoked_dict = {event_names[0]:epochs[event_names[0]].average(),
                              event_names[1]:epochs[event_names[1]].average()}

  ## plot ERP at each electrode
  if plot_electrodes:
    picks = pick_types(evoked_dict[event_names[0]].info, meg=False, eeg=True, eog=False)
    fig_zero = evoked_dict[event_names[0]].plot(spatial_colors=True,picks=picks)
    fig_zero = evoked_dict[event_names[1]].plot(spatial_colors=True,picks=picks)

  ## plot ERP in each condition on same plot
  if plot_erp:
    #find the electrode most miximal on the head (highest in z)
    picks = np.argmax([evoked_dict[event_names[0]].info['chs'][i]['loc'][2] 
              for i in range(len(evoked_dict[event_names[0]].info['chs']))])
    colors = {event_names[0]:"Red",event_names[1]:"Blue"}
    viz.plot_compare_evokeds(evoked_dict,colors=colors,
                            picks=picks,split_legend=True)

  return epochs, evoked_dict



def FeatureEngineer(epochs, model_type='NN',
                    frequency_domain=False,
                    normalization=True, electrode_median=False,
                    wavelet_decim=1,flims=(3,30),
                    f_bins=20,wave_cycles=3,
                    spect_baseline=[-1,-.5],
                    electrodes_out=[11,12,13,14,15],
                    test_split = 0.2, val_split = 0.2,
                    random_seed=1017, watermark = False):

  """
  Takes epochs object as input and settings, outputs training, test and val data
  option to use frequency or time domain
  take epochs? tfr? or autoencoder encoded object?
  """
  np.random.seed(random_seed)

  #pull event names in order of trigger number
  epochs.event_id = {'cond0':1, 'cond1':2}
  event_names = ['cond0','cond1']
  i = 0
  for key, value in sorted(epochs.event_id.items(),
                           key=lambda item: (item[1],item[0])):
    event_names[i] = key
    i += 1

  #Create feats object for output
  feats = Feats()
  feats.num_classes = len(epochs.event_id)
  feats.model_type = model_type

  if frequency_domain:
    print('Constructing Frequency Domain Features')
    f_low = flims[0]
    f_high = flims[1]
    frequencies =  np.linspace(f_low, f_high, f_bins, endpoint=True)

    ####
    ## Condition0 ##
    print('Computing Morlet Wavelets on ' + event_names[0])
    tfr0 = tfr_morlet(epochs[event_names[0]], freqs=frequencies,
                          n_cycles=wave_cycles, return_itc=False,
                          picks=electrodes_out, average=False,
                          decim=wavelet_decim)
    tfr0 = tfr0.apply_baseline(spect_baseline,mode='mean')
    #reshape data
    stim_onset = np.argmax(tfr0.times>0)
    feats.new_times = tfr0.times[stim_onset:]

    #move electrodes last
    cond0_power_out = np.moveaxis(tfr0.data[:,:,:,stim_onset:],1,3)
    # move time second
    cond0_power_out = np.moveaxis(cond0_power_out,1,2)
    ####

    ####
    ## Condition1 ##
    print('Computing Morlet Wavelets on ' + event_names[1])
    tfr1 = tfr_morlet(epochs[event_names[1]], freqs=frequencies,
                          n_cycles=wave_cycles, return_itc=False,
                          picks=electrodes_out, average=False,
                          decim=wavelet_decim)
    tfr1 = tfr1.apply_baseline(spect_baseline,mode='mean')
    #reshape data
    cond1_power_out = np.moveaxis(tfr1.data[:,:,:,stim_onset:],1,3)
    cond1_power_out = np.moveaxis(cond1_power_out,1,2) # move time second
    ####

    print('Condition one trials: ' + str(len(cond1_power_out)))
    print(event_names[1] + ' Time Points: ' + str(len(feats.new_times)))
    print(event_names[1] + ' Frequencies: ' + str(len(tfr1.freqs)))
    print('Condition zero trials: ' + str(len(cond0_power_out)))
    print(event_names[0] + ' Time Points: ' + str(len(feats.new_times)))
    print(event_names[0] + ' Frequencies: ' + str(len(tfr0.freqs)))


    #Construct X and Y
    X = np.append(cond0_power_out,cond1_power_out,0);
    Y_class = np.append(np.zeros(len(cond0_power_out)),
                        np.ones(len(cond1_power_out)),0)

    if electrode_median:
      print('Computing Median over electrodes')
      X = np.expand_dims(np.median(X,axis=len(X.shape)-1),2)

    #reshape to trials x times x variables for LSTM and NN model
    if model_type == 'NN' or model_type == 'LSTM':
      X = np.reshape(X, (X.shape[0], X.shape[1], np.prod(X.shape[2:])))

    if model_type == 'CNN3D':
      X = np.expand_dims(X,4)

    if model_type == 'AUTO' or model_type == 'AUTODeep':
      print('Auto model reshape')
      X = np.reshape(X, (X.shape[0],np.prod(X.shape[1:])))


  if not frequency_domain:
    print('Constructing Time Domain Features')

    #if using muse aux port as eeg must label it as such
    eeg_chans = pick_types(epochs.info,eeg=True,eog=False)
    #put channels last, remove eye and stim
    X = np.moveaxis(epochs._data[:,eeg_chans,:],1,2);

    #take post baseline only
    stim_onset = np.argmax(epochs.times>0)
    feats.new_times = epochs.times[stim_onset:]
    X = X[:,stim_onset:,:]

    #convert markers to class
    #requires markers to be 1 and 2 in data file?
    #This probably is not robust to other marker numbers
    Y_class = epochs.events[:,2]-1  #subtract 1 to make 0 and 1

    if electrode_median:
      print('Computing Median over electrodes')
      X = np.expand_dims(np.median(X,axis=len(X.shape)-1),2)

    ## Model Reshapes:
    # reshape for CNN
    if model_type == 'CNN':
      print('Size X before reshape for CNN: ' + str(X.shape))
      X = np.expand_dims(X,3 )
      print('Size X before reshape for CNN: ' + str(X.shape))

    # reshape for CNN3D
    if model_type == 'CNN3D':
      print('Size X before reshape for CNN3D: ' + str(X.shape))
      X = np.expand_dims(np.expand_dims(X,3),4)
      print('Size X before reshape for CNN3D: ' + str(X.shape))

    #reshape for autoencoder
    if model_type == 'AUTO' or model_type == 'AUTODeep':
      print('Size X before reshape for Auto: ' + str(X.shape))
      X = np.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
      print('Size X after reshape for Auto: ' + str(X.shape))


  #Normalize X - need to save mean and std for future test + val
  if normalization:
    print('Normalizing X')
    X = (X - np.mean(X)) / np.std(X)

  # convert class vectors to one hot Y and recast X
  Y = keras.utils.to_categorical(Y_class,feats.num_classes)
  X = X.astype('float32')

  # add watermark for testing models
  if watermark:
    X[Y[:,0]==0,0:2,] = 0
    X[Y[:,0]==1,0:2,] = 1

  # Compute model input shape
  feats.input_shape = X.shape[1:]

  # Split training test and validation data
  val_prop = val_split / (1-test_split)
  (feats.x_train,
    feats.x_test,
    feats.y_train,
    feats.y_test) = train_test_split(X, Y,
                                     test_size=test_split,
                                     random_state=random_seed)
  (feats.x_train,
   feats.x_val,
   feats.y_train,
   feats.y_val) = train_test_split(feats.x_train, feats.y_train,
                                   test_size=val_prop,
                                   random_state=random_seed)

  #compute class weights for uneven classes
  y_ints = [y.argmax() for y in feats.y_train]
  feats.class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_ints),
                                                 y_ints)

  #Print some outputs
  print('Combined X Shape: ' + str(X.shape))
  print('Combined Y Shape: ' + str(Y_class.shape))
  print('Y Example (should be 1s & 0s): ' + str(Y_class[0:10]))
  print('X Range: ' + str(np.min(X)) + ':' + str(np.max(X)))
  print('Input Shape: ' + str(feats.input_shape))
  print('x_train shape:', feats.x_train.shape)
  print(feats.x_train.shape[0], 'train samples')
  print(feats.x_test.shape[0], 'test samples')
  print(feats.x_val.shape[0], 'validation samples')
  print('Class Weights: ' + str(feats.class_weights))

  return feats





def CreateModel(feats,units=[16,8,4,8,16], dropout=.25,
                batch_norm=True, filt_size=3, pool_size=2):

  print('Creating ' +  feats.model_type + ' Model')
  print('Input shape: ' + str(feats.input_shape))


  nunits = len(units)

  ##---LSTM - Many to two, sequence of time to classes
  #Units must be at least two
  if feats.model_type == 'LSTM':
    if nunits < 2:
      print('Warning: Need at least two layers for LSTM')

    model = Sequential()
    model.add(LSTM(input_shape=(None, feats.input_shape[1]),
                   units=units[0], return_sequences=True))
    if batch_norm:
      model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout:
      model.add(Dropout(dropout))

    if len(units) > 2:
      for unit in units[1:-1]:
        model.add(LSTM(units=unit,return_sequences=True))
        if batch_norm:
          model.add(BatchNormalization())
        model.add(Activation('relu'))
        if dropout:
          model.add(Dropout(dropout))

    model.add(LSTM(units=units[-1],return_sequences=False))
    if batch_norm:
      model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout:
      model.add(Dropout(dropout))

    model.add(Dense(units=feats.num_classes))
    model.add(Activation("softmax"))


  ##---DenseFeedforward Network
  #Makes a hidden layer for each item in units
  if feats.model_type == 'NN':
    model = Sequential()
    model.add(Flatten(input_shape=feats.input_shape))

    for unit in units:
      model.add(Dense(unit))
      if batch_norm:
        model.add(BatchNormalization())
      model.add(Activation('relu'))
      if dropout:
        model.add(Dropout(dropout))

    model.add(Dense(feats.num_classes, activation='softmax'))

  ##----Convolutional Network
  if feats.model_type == 'CNN':
    if nunits < 2:
      print('Warning: Need at least two layers for CNN')
    model = Sequential()
    model.add(Conv2D(units[0], filt_size,
              input_shape=feats.input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    if nunits > 2:
      for unit in units[1:-1]:
        model.add(Conv2D(unit, filt_size, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size, padding='same'))


    model.add(Flatten())
    model.add(Dense(units[-1]))
    model.add(Activation('relu'))
    model.add(Dense(feats.num_classes))
    model.add(Activation('softmax'))

  ##----Convolutional Network
  if feats.model_type == 'CNN3D':
    if nunits < 2:
      print('Warning: Need at least two layers for CNN')
    model = Sequential()
    model.add(Conv3D(units[0], filt_size,
                     input_shape=feats.input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=pool_size, padding='same'))

    if nunits > 2:
      for unit in units[1:-1]:
        model.add(Conv3D(unit, filt_size, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=pool_size, padding='same'))


    model.add(Flatten())
    model.add(Dense(units[-1]))
    model.add(Activation('relu'))
    model.add(Dense(feats.num_classes))
    model.add(Activation('softmax'))


  ## Autoencoder
  #takes the first item in units for hidden layer size
  if feats.model_type == 'AUTO':
    encoding_dim = units[0]
    input_data = Input(shape=(feats.input_shape[0],))
    #,activity_regularizer=regularizers.l1(10e-5)
    encoded = Dense(encoding_dim, activation='relu')(input_data)
    decoded = Dense(feats.input_shape[0], activation='sigmoid')(encoded)
    model = Model(input_data, decoded)

    encoder = Model(input_data,encoded)
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = model.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))


  #takes an odd number of layers > 1
  #e.g. units = [64,32,16,32,64]
  if feats.model_type == 'AUTODeep':
    if nunits % 2 == 0:
      print('Warning: Please enter odd number of layers into units')

    half = nunits/2
    midi = int(np.floor(half))

    input_data = Input(shape=(feats.input_shape[0],))
    encoded = Dense(units[0], activation='relu')(input_data)

    #encoder decreases
    if nunits >= 3:
        for unit in units[1:midi]:
          encoded = Dense(unit, activation='relu')(encoded)

    #latent space
    decoded = Dense(units[midi], activation='relu')(encoded)

    #decoder increses
    if nunits >= 3:
      for unit in units[midi+1:-1]:
        decoded = Dense(unit, activation='relu')(decoded)

    decoded = Dense(units[-1], activation='relu')(decoded)

    decoded = Dense(feats.input_shape[0], activation='sigmoid')(decoded)
    model = Model(input_data, decoded)

    encoder = Model(input_data,encoded)
    encoded_input = Input(shape=(units[midi],))





  if feats.model_type == 'AUTO' or feats.model_type == 'AUTODeep':
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='mean_squared_error')



  if ((feats.model_type == 'CNN') or
      (feats.model_type == 'CNN3D') or
      (feats.model_type == 'LSTM') or
      (feats.model_type == 'NN')):

    # initiate adam optimizer
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    # Let's train the model using RMSprop
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    encoder = []


  model.summary()

  return model, encoder


def TrainTestVal(model, feats, batch_size=2, train_epochs=20, show_plots=True):
  print('Training Model:')
  # Train Model
  if feats.model_type == 'AUTO' or feats.model_type == 'AUTODeep':
    print('Training autoencoder:')

    history = model.fit(feats.x_train, feats.x_train,
                        batch_size = batch_size,
                        epochs=train_epochs,
                        validation_data=(feats.x_val,feats.x_val),
                        shuffle=True,
                        verbose=True,
                        class_weight=feats.class_weights
                       )

    # list all data in history
    print(history.history.keys())

    if show_plots:
      # summarize history for loss
      plt.semilogy(history.history['loss'])
      plt.semilogy(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.show()

  else:
    history = model.fit(feats.x_train, feats.y_train,
              batch_size=batch_size,
              epochs=train_epochs,
              validation_data=(feats.x_val, feats.y_val),
              shuffle=True,
              verbose=True,
              class_weight=feats.class_weights
              )

    # list all data in history
    print(history.history.keys())

    if show_plots:
      # summarize history for accuracy
      plt.plot(history.history['acc'])
      plt.plot(history.history['val_acc'])
      plt.title('model accuracy')
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.show()
      # summarize history for loss
      plt.semilogy(history.history['loss'])
      plt.semilogy(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.show()


    # Test on left out Test data
    score, acc = model.evaluate(feats.x_test, feats.y_test,
                                batch_size=batch_size)
    print(model.metrics_names)
    print('Test loss:', score)
    print('Test accuracy:', acc)

    # Build a dictionary of data to return
    data = {}
    data['score'] = score
    data['acc'] = acc

    return model, data