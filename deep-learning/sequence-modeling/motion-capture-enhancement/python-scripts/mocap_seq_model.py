# LIBRARIES IMPORT

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seglearn as seg
import tensorflow as tf

from seglearn.transform import SegmentX, SegmentXY, last

# SETTINGS

#   Main Directory
DATADIR = "C:/Users/bdour/Documents/Work/Toronto/Sunnybrook/ACL Injury Screening/Data/Processed/Machine learning/Enhancement"

#   Filenames
#       Note: for normalized data (varying between 0 and 1), edit filename below following this example: train_input_name = 'K_train_jc_input_X_norm.csv'
train_input_name = 'K_train_R_jc_input_X.csv'
train_target_name = 'V_train_RK_fl_target_y.csv'
val_input_name = 'K_val_R_jc_input_X.csv'
val_target_name = 'V_val_RK_fl_target_y.csv'
#       Trained model export
mod_export_name = 'seq2p_RK_fl_step1_20_b32_lr001.h5'

#   Trial segmentation parameters
step_size = 1
seg_width = 20

#   Training parameters
batch_size = 32
epochs = 200
#   Optimizer (choose between following options)
#       Adam: keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#       RMSprop: keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
opt = tf.keras.optimizers.RMSprop(lr=0.001)

# PATHS DEFINITION

train_input_path = os.path.join(DATADIR, 'Training data/' + train_input_name)
train_target_path = os.path.join(DATADIR, 'Training data/' + train_target_name)
val_input_path = os.path.join(DATADIR, 'Validation data/' + val_input_name)
val_target_path = os.path.join(DATADIR, 'Validation data/' + val_target_name)

# DATA IMPORT

#   Import CSV files
df_train_X = pd.read_csv(train_input_path)
df_train_y = pd.read_csv(train_target_path)
df_val_X = pd.read_csv(val_input_path)
df_val_y = pd.read_csv(val_target_path)

#   Isolate numerical values
train_X = df_train_X.values
train_y = df_train_y.values
val_X = df_val_X.values
val_y = df_val_y.values

print('-> Data imported')

# DATA PROCESSING

#   Reshaped data initialization
train_Xr, train_yr, val_Xr, val_yr = [], [], [], []

#   Reshape data into a (m, n, l) shaped array, where m represents the number of samples (= trials), n the length of each signal (changes across trials), and l the number of channels (= number of columns)

#       Training data
for i in np.unique(train_X[:,1]):            # Note: here, we look at column with index = 1, which is the 2nd column of the CSV file, which is the unique trial ID
    Xi = train_X[train_X[:,1] == i, 2:]      # Note: here, we take all columns from index = 2 and after, which correspond to the 3rd column and after, which corresponds to all the channels
    yi = train_y[train_y[:,1] == i, 2]
    train_Xr.append(Xi)
    train_yr.append(yi)
    if len(Xi) != len(yi):
        raise ValueError("Different lengths between X ", len(Xi), " and Y ", len(yi), " for trial ID: ", i)

#       Validation data
for i in np.unique(val_X[:,1]):
    Xi = val_X[val_X[:,1] == i, 2:]
    yi = val_y[val_y[:,1] == i, 2]
    val_Xr.append(Xi)
    val_yr.append(yi)
    if len(Xi) != len(yi):
        raise ValueError("Different lengths between X ", len(Xi), " and Y ", len(yi), " for trial ID: ", i)

#   Calculate the length of each trial
train_lens = [len(x) for x in train_Xr]
val_lens = [len(x) for x in val_Xr]

#   Convert reshaped lists into array
train_Xr, train_yr, val_Xr, val_yr = np.array(train_Xr), np.array(train_yr), np.array(val_Xr), np.array(val_yr)

#   Segment each trial into multiple segments, with only one target value per segment (= last value of y for the corresponding segment)
seg = SegmentXY(step=step_size, width=seg_width, order='C', y_func=last)
train_Xs, train_ys, _ = seg.transform(train_Xr, train_yr)
val_Xs, val_ys, _ = seg.transform(val_Xr, val_yr)

print('-> Data processed')
print('     Training')
print('         Input data shape:              ', np.shape(train_Xs))
print('         Target data shape:             ', np.shape(train_ys))
print('     Validation')
print('         Input data shape:              ', np.shape(val_Xs))
print('         Target data shape:             ', np.shape(val_ys))

# TIME SERIES REGRESSION using RECURRENT LONG-SHORT TERM MEMORY (LSTM) NEURAL NETWORK
#   Based on TensorFlow tutorial for time series forecasting (https://www.tensorflow.org/tutorials/structured_data/time_series)

print('')
print('-> TIME SERIES REGRESSION using LSTM RNN')
print('')

#   Model definition
model = tf.keras.Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, input_shape=np.shape(train_Xs))))
model.add(tf.keras.layers.Dense(1))

print('-> Model defined')

#   Compile model
model.compile(optimizer=opt, loss='mae')

print('-> Model compiled')

#   Train model
history = model.fit(train_Xs, train_ys, batch_size=batch_size, epochs=epochs, validation_data=(val_Xs, val_ys))

print('-> Model trained')

#   Save model
model.save(os.path.join("C:/Users/bdour/Documents/Work/Toronto/Sunnybrook/ACL Injury Screening/Data/Processed/Machine learning/Enhancement", mod_export_name))

# PLOT TRAINING HISTORY

plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], c='royalblue', ls='--', label='Training loss')
plt.plot(history.history['val_loss'], c='chocolate', ls='--', label='Validation loss')
plt.ylabel('Loss [Mean absolute error]')
plt.xlabel('Epoch')
plt.title('Training history')
plt.legend()
plt.show()

#   Apply model to make prediction
predic = model.predict(val_Xs, batch_size=batch_size)

#   Display true label and prediction
plt.figure(figsize=(6, 4))
plt.plot(val_ys, c='royalblue', label='Actual')
plt.plot(predic, c='chocolate', ls='--', label='Predicted')
plt.ylabel('Right knee flexion')
plt.xlabel('segment')
plt.title('Prediction')
plt.legend()
plt.show()


print('')
print('--------------')
print('CODE COMPLETED')
print('--------------')
