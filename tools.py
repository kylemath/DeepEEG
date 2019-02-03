def preprocess(raw, event_id, plot_psd=True, filter_data=True, 
               eeg_filter_highpass=1, plot_events=True, epoch_time=(-1,2), 
               baseline=(-.2,0), rej_thresh_uV=200,
               epoch_decim=1, plot_electrodes=True,
               plot_erp=True):

  from mne import find_events
  from mne import pick_types, viz, Epochs

  sfreq = raw.info['sfreq']
  nsfreq = sfreq/epoch_decim #create new output freq for after epoch or wavelet decim
  tmin=epoch_time[0] 
  tmax=epoch_time[1] 
  eeg_filter_lowpass = nsfreq/2.5  #lower to avoid aliasing from decim

  #pull event names in order of trigger number
  event_names = ['cond0','cond1']
  i = 0
  for key, value in sorted(event_id.iteritems(), key=lambda (k,v): (v,k)):
    event_names[i] = key
    i += 1

  #Filtering

  if filter_data:             
    print('Filtering Data')
    raw.filter(eeg_filter_highpass,eeg_filter_lowpass, 
               method='iir', verbose='WARNING' )
  
  if plot_psd:
    raw.plot_psd(fmin=eeg_filter_highpass, fmax=nsfreq/2 ) 
   
  #artifact rejection
  rej_thresh = rej_thresh_uV*1e-6

  #Epoching
  events = find_events(raw,shortest_event=1)
  color = {1: 'red', 2: 'black'}

  #plot event timing
  if plot_events:
    viz.plot_events(events, sfreq, raw.first_samp, color=color,
                        event_id=event_id)

  #Constructevents
  epochs = Epochs(raw, events=events, event_id=event_id, 
                  tmin=tmin, tmax=tmax, baseline=baseline, 
                  preload=True,reject={'eeg':rej_thresh},
                  verbose=False, decim=epoch_decim)
  print('sample drop %: ', (1 - len(epochs.events)/len(events)) * 100)
   
  if plot_electrodes or plot_erp:
    evoked_zero = epochs[event_names[0]].average()
    evoked_one = epochs[event_names[1]].average()  
  
  ## plot ERP at each electrode
  if plot_electrodes:
    pick = pick_types(epochs.info, meg=False, eeg=True, eog=False)
    fig_zero = evoked_zero.plot(spatial_colors=True, picks=pick)
    fig_zero = evoked_one.plot(spatial_colors=True, picks=pick)

  ## plot ERP in each condition on same plot
  if plot_erp:
    evoked_dict = dict()
    evoked_dict['eventZero'] = evoked_zero
    evoked_dict['eventOne'] = evoked_one
    colors = dict(eventZero="Red", eventOne="Blue")
    pick = [0,1,2,3]
    viz.plot_compare_evokeds(evoked_dict, picks=pick, colors=colors,
                                 split_legend=True)

  return epochs





def FeatureEngineer(epochs, model_type='NN',
                    frequency_domain=0,
                    normalization=True, electrode_median=False,
                    wavelet_decim=1,flims=(3,30),
                    f_bins=20,wave_cycles=6,
                    spect_baseline=[-1,-.5],
                    electrodes_out=[11,12,13,14,15],
                    test_split = 0.2, val_split = 0.2,
                    random_seed=1017):
  

  class Feats:
    def __init__(self,num_classes,class_weights,input_shape,x_train,y_train,x_test,y_test,x_val,y_val):
      self.num_classes = 2
      self.class_weights = [1., 1.]
      self.input_shape = 16
      self.x_train = 1
      self.y_train = 1
      self.x_test = 1
      self.y_test = 1
      self.x_val = 1
      self.y_val = 1


  #Takes epochs object as input and settings, outputs training, test and val data
  #option to use frequency or time domain
  #take epochs? tfr? or autoencoder encoded object?
  
  import numpy as np
  import keras
  from sklearn.model_selection import train_test_split
  from sklearn.utils import class_weight
  from mne.time_frequency import tfr_morlet
  from DeepEEG.utils import factors
  
  #Training Settings
  
  #pull event names in order of trigger number
  event_names = ['cond0','cond1']
  i = 0
  for key, value in sorted(epochs.event_id.iteritems(), key=lambda (k,v): (v,k)):
    event_names[i] = key
    i += 1
  
  test = len(epochs.event_id)
  feats = Feats()
  feats.num_classes = len(epochs.event_id)
  np.random.seed(random_seed)

  if frequency_domain:
    print('Constructing Frequency Domain Features')
    f_low = flims[0]
    f_high = flims[1]
    frequencies =  np.linspace(f_low, f_high, f_bins, endpoint=True)

    
    
    ## Condition0 ##
    print('Computing Morlet Wavelets on ' + event_names[0])
    tfr0 = tfr_morlet(epochs[event_names[0]], freqs=frequencies, 
                          n_cycles=wave_cycles, return_itc=False,
                          picks=electrodes_out,average=False,decim=wavelet_decim)
    tfr0 = tfr0.apply_baseline(spect_baseline,mode='mean')
    stim_onset = np.argmax(tfr0.times>0)
    feats.new_times = tfr0.times[stim_onset:]
    #reshape data
    cond0_power_out = np.moveaxis(tfr0.data[:,:,:,stim_onset:],1,3) #move electrodes last
    cond0_power_out = np.moveaxis(cond0_power_out,1,2) # move time second
    #cond0_power_out[:,0:5,0:5,:] = 0 #for testing model add mark to image
    

    ## Condition1 ##
    print('Computing Morlet Wavelets on ' + event_names[1])
    tfr1 = tfr_morlet(epochs[event_names[1]], freqs=frequencies, 
                          n_cycles=wave_cycles, return_itc=False,
                          picks=electrodes_out,average=False,decim=wavelet_decim)
    tfr1 = tfr1.apply_baseline(spect_baseline,mode='mean')   
    #reshape data
    cond1_power_out = np.moveaxis(tfr1.data[:,:,:,stim_onset:],1,3)
    cond1_power_out = np.moveaxis(cond1_power_out,1,2) # move time second
    #cond1_power_out[:,0:5,0:5,:] = 1 #for testing model add mark to image

    
    
    
    print('Condition one trials: ' + str(len(cond1_power_out)))    
    print(event_names[1] + ' Time Points: ' + str(len(new_times)))
    print(event_names[1] + ' Frequencies: ' + str(len(tfr1.freqs)))
    print('Condition zero trials: ' + str(len(cond0_power_out)))
    print(event_names[0] + ' Time Points: ' + str(len(new_times)))
    print(event_names[0] + ' Frequencies: ' + str(len(tfr0.freqs)))

    
    #Construct X
    X = np.append(cond0_power_out,cond1_power_out,0);

    #reshape to trials x times x variables for LSTM and NN model
    if model_type != 'CNN':
      X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2] * X.shape[3]),order='F')
    
    if model_type == 'AUTO':
      print('Auto model reshape')
      X = np.reshape(X, (X.shape[0],X.shape[1]*X.shape[2]*X.shape[3]))
      

    #Append Data
    Y_class = np.append(np.zeros(len(cond0_power_out)), np.ones(len(cond1_power_out)),0)


  if not frequency_domain:
    print('Constructing Time Domain Features')

    X = np.moveaxis(epochs._data[:,:-3,:],1,2); #put channels last, remove eye and stim

    #take post baseline only
    stim_onset = np.argmax(epochs.times>0)
    new_times = epochs.times[stim_onset:]
    X = X[:,stim_onset:,:]
    Y_class = epochs.events[:,2]-1  #subtract 1 to make 0 and 1

    # reshape for CNN, factor middle dimensions
    if model_type == 'CNN' and not frequency_domain:
      all_factors = factors(X.shape[1])
      X = np.reshape(X, (X.shape[0], int(X.shape[1]/all_factors[2]), all_factors[2], X.shape[2]),order='F')
      
    if electrode_median:
      print('Computing Median over electrodes')
      X = np.expand_dims(np.median(X,axis=len(X.shape)-1),2)
        
      
      
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

  # Split training test and validation data 
  val_prop = val_split / (1-test_split)
  feats.x_train, feats.x_test, feats.y_train, feats.y_test = train_test_split(X, Y, test_size=test_split,random_state=random_seed) 
  feats.x_train, feats.x_val, feats.y_train, feats.y_val = train_test_split(feats.x_train, feats.y_train, test_size=val_prop, random_state=random_seed)

  # Compute model input shape
  feats.input_shape = X.shape[1:]
  
  #compute class weights for uneven classes
  feats.y_ints = [y.argmax() for y in feats.y_train]
  feats.class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_ints),
                                                 y_ints)
  
  #Print some outputs
  print('Combined X Shape: ' + str(X.shape))
  print('Combined Y Shape: ' + str(feats.Y_class.shape))
  print('Y Example (should be 1s & 0s): ' + str(feats.Y_class[0:10]))
  print('X Range: ' + str(np.min(X)) + ':' + str(np.max(X)))
  print('Input Shape: ' + str(feats.input_shape))
  print('x_train shape:', feats.x_train.shape)
  print(feats.x_train.shape[0], 'train samples')
  print(feats.x_test.shape[0], 'test samples')
  print(feats.x_val.shape[0], 'validation samples')
  print('Class Weights: ' + str(feats.class_weights))

  return feats





def CreateModel(Feats,model_type='NN',batch_size=1):
  print('Creating ' +  model_type + ' Model')
  
  import keras
  from keras.models import Sequential, Model
  from keras.layers import Dense, Dropout, Activation, Input
  from keras.layers import Flatten, Conv2D, MaxPooling2D, LSTM

  
  ##---LSTM - Many to two, sequence of time to classes
  if model_type == 'LSTM':
    units = [Feats.input_shape[1], 100, 100, 100, 100, Feats.num_classes]
    model = Sequential()
    model.add(LSTM(input_shape=(None, units[0]) ,units=units[1], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units[2],return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units[3],return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units[4],return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=units[5]))    
    model.add(Activation("softmax"))
    
    
  ##---DenseFeedforward Network
  if model_type == 'NN':
    from keras.layers import BatchNormalization
    model = Sequential()
    model.add(Flatten())
    
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))
    
    model.add(Dense(Feats.num_classes, activation='softmax'))

  ##----Convolutional Network                  
  if model_type == 'CNN':
    model = Sequential()
    model.add(Conv2D(10, (3, 3), input_shape=Feats.input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(Feats.num_classes))
    model.add(Activation('softmax'))
    
  if model_type == 'AUTO': 
    encoding_dim = 16
    input_data = Input(shape=(Feats.input_shape[0],))
    encoded = Dense(encoding_dim, activation='relu')(input_data) #,activity_regularizer=regularizers.l1(10e-5)
    decoded = Dense(Feats.input_shape[0], activation='sigmoid')(encoded)
    model = Model(input_data, decoded)
    
    
    encoder = Model(input_data,encoded)
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = model.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    
    
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, 
                                epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='mean_squared_error')
    
 
  
  if model_type == 'AUTODeep': 
    units = [128,64,32,16,32,64,128]
    input_data = Input(shape=(Feats.input_shape[0],))
    encoded = Dense(units[0], activation='relu')(input_data)
    encoded = Dense(units[1], activation='relu')(encoded)
    encoded = Dense(units[2], activation='relu')(encoded)
    encoded = Dense(units[3], activation='relu')(encoded)
    decoded = Dense(units[4], activation='relu')(encoded) 
    decoded = Dense(units[5], activation='relu')(encoded) 
    decoded = Dense(units[6], activation='relu')(decoded)
    decoded = Dense(Feats.input_shape[0], activation='sigmoid')(decoded)
    model = Model(input_data, decoded)
        
    encoder = Model(input_data,encoded)
    encoded_input = Input(shape=(units[2],))
    
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, 
                                epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='mean_squared_error')
 
  
  
  
  if model_type == 'CNN' or model_type == 'LSTM' or model_type == 'NN':
    # initiate adam optimizer
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, 
                                epsilon=None, decay=0.0, amsgrad=False)
    # Let's train the model using RMSprop
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy']) 
    encoder = []
    
  return model, encoder






def traintestval(model,batch_size=1,train_epochs=20,model_type='NN'):
  class_weights,x_train,x_test,x_val,y_train,y_test,y_val,
  print('Training Model:')
  import matplotlib.pyplot as plt

  #Train Model
  if model_type == 'AUTO' or model_type == 'AUTODeep':
    print('Training autoencoder:')
   
    history = model.fit(Feats.x_train, Feats.x_train,
                        batch_size = batch_size,
                        epochs=train_epochs,
                        validation_data=(Feats.x_val,Feats.x_val),
                        shuffle=True,
                        verbose=True,
                        class_weight=Feats.class_weights
                       )
    
    # list all data in history
    print(history.history.keys())
    
    # summarize history for loss
    plt.semilogy(history.history['loss'])
    plt.semilogy(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

  else:
    
    history = model.fit(Feats.x_train, Feats.y_train,
              batch_size=batch_size,
              epochs=train_epochs,
              validation_data=(Feats.x_val, Feats.y_val),
              shuffle=True,
              verbose=True,
              class_weight=Feats.class_weights
              )

    # list all data in history
    print(history.history.keys())
    
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
    score, acc = model.evaluate(Feats.x_test, Feats.y_test, batch_size=batch_size)
    print(model.metrics_names)
    print('Test loss:', score)
    print('Test accuracy:', acc)

  #Summarize
  model.summary()

