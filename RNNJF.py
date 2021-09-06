import os
import numpy as np
import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d as plt3d
import pandas as pd
# import math
# import copy
# import pickle
from keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Layer, TimeDistributed, Dropout
from keras.layers import Dense, Input, Masking, LSTM
from sklearn.preprocessing import MinMaxScaler, RobustScaler
# from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import argparse

def load_data(DF, remove_dirtrack, add_dirtrack, features=['d0','z0','phi','theta','q/p']):
  all_features = np.array(['d0','z0','phi','theta','q/p','x_o','y_o','z_o','x_p','y_p','z_p'])
  print("reading the datafile")
  bjets_DF = pd.read_pickle(DF) #"./bjets_IPonly_abs_10um_errs.pkl")
  print("loading tracks")
  X = get_tracks(bjets_DF,features)
  print("preprocessing the data")
  if remove_dirtrack: X = remove_direction_track(X,len(features))
  nodirtrack = remove_dirtrack or not add_dirtrack
  if args.order_by_feature is not None:
    if args.reverse: print("ordering by decreasing {}".format(args.order_by_feature))
    else: print("ordering by increasing {}".format(args.order_by_feature))
    fidx=np.where(all_features == args.order_by_feature)[0][0]
    X = order_by_feature(X, nodirtrack, args.reverse, feature=fidx)
  elif args.use_custom_order:
    raise NotImplemented()
  elif args.no_reorder:
    print("No reorder performed")
  else:
    print("using random order")
    X = order_random(X)
  X = scale_features(X,features)
  X = np.nan_to_num(X)
  y = bjets_DF[['secVtx_x', 'secVtx_y', 'secVtx_z', 'terVtx_x', 'terVtx_y', 'terVtx_z']].values
  y = y * 1000  # change units of vertices from m to mm, keep vals close to unity
  return X, y

def get_tracks(bjets_DF,features):
  ntracks=30
  all_features = np.array(['d0','z0','phi','theta','q/p','x_o','y_o','z_o','x_p','y_p','z_p'])
  nfeatures = len(features)
  feature_idx = [np.where(all_features == i)[0][0] for i in np.array(features)]
  trks = np.zeros((len(bjets_DF), ntracks, nfeatures))
  for i in range(len(bjets_DF)):
    trks[i] = np.array([bjets_DF['tracks'][i]])[:, :, feature_idx]
    # if add_dirtrack:
    #   ...
  return trks

def order_by_feature(X, nodirtrack, reverse, feature=0):
  if nodirtrack:
    print("no direction track, ordering by feature {}".format(feature))
    for i, jet in enumerate(X[:, 0:]):
      nan_ind = np.where(np.isnan(X[i]))[0][0]
      if reverse: X[i, 0:nan_ind] = jet[np.abs(jet[:nan_ind, 0]).argsort()[::-1]]
      else: X[i, 0:nan_ind] = jet[np.abs(jet[:nan_ind, 0]).argsort()]
  else:
    for i, jet in enumerate(X[:, 1:]):
      nan_ind = np.where(np.isnan(X[i]))[0][0]
      if reverse: X[i, 1:nan_ind] = jet[np.abs(jet[:nan_ind, 0]).argsort()[::-1]]
      else: X[i, 1:nan_ind] = jet[np.abs(jet[:nan_ind, 0]).argsort()]
  return X

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
  for i, feature in enumerate(features):
    var_to_scale = Xscaled[:, :, i].reshape(300000 * 30)
    var_to_scale = var_to_scale.reshape(-1, 1)
    if feature in ['d0','z0','q/p']:
      print("Robust Scaling: {}".format(feature))
      scaler = RobustScaler()  # maybe have another look at this case, it seems to skew d0 quite a lot
    else:
      print("MinMax Scaling: {}".format(feature))
      scaler = MinMaxScaler([-1, 1])
    scaler.fit(var_to_scale)
    scaled_var = scaler.transform(var_to_scale)
    Xscaled[:, :, i] = scaled_var.reshape(300000, 30)
  return Xscaled

def split_train_test(X,y,split=280000,seed=None):
  # X_train = X[:split]
  # X_test = X[split:]
  # y_train = y[:split]
  # y_test = y[split:]
  ts = (300000-split)/300000
  if seed==None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts)#, random_state=42)
  else:
    print("train/test split with seed {}".format(seed))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=seed)
  print(X_train.shape)
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

def plot_loss(myRNN_hist,out):
  epochs = np.arange(1, len(myRNN_hist.history['loss']) + 1)

  plt.plot(epochs, myRNN_hist.history['loss'], label='training')
  plt.plot(epochs, myRNN_hist.history['val_loss'], label='validation')
  plt.xlabel('epochs', fontsize=14)
  plt.ylabel('MSE loss', fontsize=14)
  plt.legend()
  plt.savefig(os.path.join(out,"loss.png"))

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--remove_dirtrack", action="store_true", default=False,
                      help="remove direction track, only for old files")
  parser.add_argument("--add_dirtrack", action="store_true", default=False,
                      help="add direction track, only for new files")
  parser.add_argument("--input", type=str, default="../bjets_newminerrs.pkl", help="input datafile (pickle)")
  parser.add_argument("--features", type=list, default=['d0', 'z0', 'phi', 'theta', 'q/p'],
                      help="list of features to use (list of strings)")
  parser.add_argument("--epochs", type=int, default=100, help="max n of epochs")
  parser.add_argument("--nHidden", type=int, default=300, help="n of nodes in hidden layer (RNN)")
  parser.add_argument("--nDense", type=int, default=40, help="n of nodes in FC layer after RNN")
  parser.add_argument("--out", type=str, default='out/',
                      help="output directory for weights of trained network and loss curves")
  parser.add_argument("--split", type=int, default='20000', help="Default n of training samples used (max 300k)")
  parser.add_argument("--order_by_feature", type=str, default=None,
                      help="Order by the defined feature ('d0','z0',etc.)")
  parser.add_argument("--reverse", action="store_true", default=False, help="In case of order by feature, order by decreasing value if reverse is true")
  parser.add_argument("--use_custom_order", action="store_true", default=False, help="use custom ordering")
  parser.add_argument("--no_reorder", action="store_true", default=False, help="No re-ordering done (uses order particles made)")
  parser.add_argument("--trial", type=int, default=1, help="trial number, fixes traintestsplit seed")
  parser.add_argument("--loss", type=str, default="mae", help="the training loss function ['mse','mae']")
  args = parser.parse_args()
  
  if args.use_custom_order and args.no_reorder:
    raise ValueError("incompatible arguments")
  if args.order_by_feature is not None and args.no_reorder:
    raise ValueError("incompatible arguments")
  if args.order_by_feature is not None and args.use_custom_order:
    raise ValueError("incompatible arguments")

  if not os.path.exists(args.out):
    os.makedirs(args.out)
  if not os.path.exists(os.path.join(args.out,"trial{}".format(args.trial))):
    os.makedirs(os.path.join(args.out,"trial{}".format(args.trial)))
  out_folder = os.path.join(args.out,"trial{}".format(args.trial))
  datafile = args.input
  model_savefile = os.path.join(out_folder,'myRNN_weights.h5')
  if "new" in datafile:
    print("no dirtrack to remove")
    remove_dirtrack = False
    # option to add dirtrack using jet information
    add_dirtrack = args.add_dirtrack
  else:
    remove_dirtrack = args.remove_dirtrack
    add_dirtrack = False

  features = args.features
  X, y = load_data(datafile,remove_dirtrack,add_dirtrack,features=features)
  X_train, X_test, y_train, y_test = split_train_test(X, y, split=args.split, seed=args.trial)
  nHidden = args.nHidden
  nDense = args.nDense
  nJets, nTrks, nFeatures = X_train.shape
  nOutputs = y.shape[1]
  myRNN = get_RNNJF(nJets, nTrks, nFeatures, nOutputs, nHidden, nDense)
  myRNN.compile(loss=args.loss, optimizer='adam',
                metrics=['mae','mse'])
  myRNN_mChkPt = ModelCheckpoint(model_savefile, monitor='val_loss', verbose=True,
                                 save_best_only=True,
                                 save_weights_only=True)
  earlyStop = EarlyStopping(monitor='val_loss', verbose=True, patience=10)
  nEpochs = args.epochs
  print("fitting to training data...")
  myRNN_hist = myRNN.fit(X_train, y_train, epochs=nEpochs, batch_size=256,validation_split=0.20,
                  callbacks=[earlyStop, myRNN_mChkPt],)
  np.save(os.path.join(out_folder,'history.npy'), myRNN_hist.history)
  plot_loss(myRNN_hist,out_folder)
