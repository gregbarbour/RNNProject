import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import pandas as pd
import math
import copy
import pickle
from keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Layer, TimeDistributed, Dropout
from keras.layers import Dense, Input, Masking, LSTM
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error

def load_data(DF, remove_dirtrack, features=['d0','z0','phi','theta','q/p']):
  ntracks=30
  all_features = np.array(['d0','z0','phi','theta','q/p','x_o','y_o','z_o','x_p','y_p','z_p'])
  nfeatures = len(features)
  feature_idx = [np.where(all_features == i)[0][0] for i in np.array(features)]
  print("reading the datafile")
  bjets_DF = pd.read_pickle(DF) #"./bjets_IPonly_abs_10um_errs.pkl")
  trks = np.zeros((len(bjets_DF), ntracks, nfeatures))
  print("loading tracks")
  for i in range(len(bjets_DF)):
    trks[i] = np.array([bjets_DF['tracks'][i]])[:, :, feature_idx]
    # if add_dirtrack:
    #   ...

  X = trks
  print("preprocessing the data")
  if remove_dirtrack: X = remove_direction_track(X,len(features))
  nodirtrack = remove_dirtrack #or not add_dirtrack
  X = order_random(X)
  X = scale_features(X,features)
  X = np.nan_to_num(X)
  y = bjets_DF[['secVtx_x', 'secVtx_y', 'secVtx_z', 'terVtx_x', 'terVtx_y', 'terVtx_z']].values
  y = y * 1000  # change units of vertices from m to mm, keep vals close to unity
  return X, y

def order_by_feature(X, nodirtrack, feature=0):
  Xordered = np.nan_to_num(X)
  if nodirtrack:
    for i, jet in enumerate(Xordered[:, 0:]):
      Xordered[i, 0:] = jet[np.abs(jet[:, feature]).argsort()[::-1]]
  else:
    for i, jet in enumerate(Xordered[:, 1:]):
      Xordered[i, 1:] = jet[np.abs(jet[:, feature]).argsort()[::-1]]

def order_random(X):
  for i in range(len(X)):
    nan_ind = np.where(np.isnan(X[i]))[0][0]
    np.random.shuffle(X[i][0:nan_ind - 1])
  return X

def remove_direction_track(X,nfeatures=5):
  X[:, 0] = np.full((300000, nfeatures), np.nan)
  X_removed_dirtrk = np.roll(X[:], -1, axis=1)
  return X_removed_dirtrk

def scale_features(X,features):
  nfeatures=len(features)
  Xscaled = X
  for track_variable in range(nfeatures):
    var_to_scale = Xscaled[:, :, track_variable].reshape(300000 * 30)
    var_to_scale = var_to_scale.reshape(-1, 1)
    if (track_variable == 0):
      print((track_variable == 0))
      print(track_variable)
      scaler = RobustScaler()  # maybe have another look at this case, it seems to skew d0 quite a lot
    elif (track_variable == 4):
      print((track_variable == 4))
      print(track_variable)
      scaler = RobustScaler()
    elif (track_variable == 1):
      print(track_variable)
      scaler = RobustScaler()
    else:
      scaler = MinMaxScaler([-1, 1])
    scaler.fit(var_to_scale)
    scaled_var = scaler.transform(var_to_scale)
    Xscaled[:, :, track_variable] = scaled_var.reshape(300000, 30)
    return Xscaled

def split_train_test(X,y,split=280000):
  X_train = X[:split]
  X_test = X[split:]
  y_train = y[:split]
  y_test = y[split:]
  return X_train, X_test, y_train, y_test

def get_RNNJF(nJets, nTrks, nFeatures, nOutputs,   nHidden = 300,   nDense = 40):
  trk_inputs = Input(shape=(nTrks, nFeatures), name="Trk_inputs")
  masked_input = Masking()(trk_inputs)

  # Feed this merged layer to an RNN
  lstm = LSTM(nHidden, return_sequences=False, name='LSTM')(masked_input)
  dpt = Dropout(rate=0.2)(lstm)  # this is a very high dropout rate, reduce it

  my_inputs = trk_inputs

  # Fully connected layer: This will convert the output of the RNN to our vtx postion predicitons
  FC = Dense(nDense, activation='relu', name="Dense")(dpt)  # is relu fine here? i think so...

  # Ouptut layer. Sec and Ter Vtx. No activation as this is a regression problem
  output = Dense(nOutputs, name="Vertex_Predictions")(FC)

  myRNN = Model(inputs=my_inputs, outputs=output)
  print(myRNN.summary())
  return myRNN

if __name__ == "__main__":

  datafile = "./bjets_IPonly_abs_10um_errs.pkl"
  model_savefile = 'myRNN_weights.h5'
  if "new" in datafile:
    remove_dirtrack = False
    # option to add dirtrack using jet information
    # add_dirtrack = True
  else:
    remove_dirtrack = True
    # add_dirtrack = False

  features = ['d0', 'z0', 'phi', 'theta', 'q/p']
  X, y = load_data(datafile,remove_dirtrack,features=features)
  X_train, X_test, y_train, y_test = split_train_test(X, y)
  nHidden = 300
  nDense = 40
  nJets, nTrks, nFeatures = X_train.shape
  nOutputs = y.shape[1]
  myRNN = get_RNNJF(nJets, nTrks, nFeatures, nOutputs, nHidden, nDense)
  myRNN.compile(loss='mean_absolute_error', optimizer='adam',
                metrics=['mae'])
  myRNN_mChkPt = ModelCheckpoint(model_savefile, monitor='val_loss', verbose=True,
                                 save_best_only=True,
                                 save_weights_only=True)
  earlyStop = EarlyStopping(monitor='val_loss', verbose=True, patience=10)
  nEpochs = 10
  print("fitting to training data...")
  myRNN_hist = myRNN.fit(X_train, y_train, epochs=nEpochs, batch_size=256,validation_split=0.20,
                  callbacks=[earlyStop, myRNN_mChkPt],)

