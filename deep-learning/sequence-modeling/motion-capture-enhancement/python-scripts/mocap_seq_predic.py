# LIBRARIES IMPORT

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seglearn as seg
import tensorflow as tf
import pylab
import copy

from scipy.stats import linregress

# DEPENDENCIES

def ICC_2way_mixed(data):
    '''
    Calculate the Intraclass Correlation Coefficient (ICC) using the Two-way Mixed Model for Case 3* defined by Patrick E. Shrout and Joseph L. Fleiss. “Intraclass Correlations: Uses in assessing rater reliability.” Psychological Bulletin 86.2 (2979): 420-428
        *In Case 3, each target/subject/observation is rated by each of the same m observers/judges/methods, who are the only observers/judges/methods of interest.
    Input:
        data: mxn array where m is the number of rows (each row is a measurement/observation/subject) and where n is the number of observers/judges/methods.
    Output:
        ICC: intraclass correlation coeeficient (3,1)
        df_m: number of degrees of freedom (df) between observers/judges/methods
        df_n: number of degrees of freedom (df) between measurements/observations/subjects
        F_stat: F-Statistic - session effect (calculated as the ratio between the vartiation between sample means and the variation within samples - i.e. ratio of two quantities that are expected to be roughly equal under the null hypothesis)
        var_obs: variance between measurements/observations/subjects
        MSE: mean squared error (calculated as the sum of squared error divided by the number of degrees of freedom between measurements/observations/subjects: SSE/df_n)
    Dependencies:
        None
    '''

    # Compute data shape and degrees of freedom
    [num_n, num_m] = data.shape
    df_m = num_m - 1
    df_n0 = num_n - 1
    df_n = df_n0 * df_m

    # Compute the sum of distance to the mean
    mean_data = np.mean(data)
    sum_dist_mean = ((data - mean_data)**2).sum()

    # Create the design matrix for the different levels
    x = np.kron(np.eye(num_m), np.ones((num_n, 1)))
    x0 = np.tile(np.eye(num_n), (num_m, 1))
    X = np.hstack([x, x0])

    # Computer the Sum of Squared Error
    predicted_data = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), data.flatten('F'))
    residuals = data.flatten('F') - predicted_data
    SSE = (residuals**2).sum()
    residuals.shape = data.shape

    # Compute the Mean Squared Error (MSE)
    MSE = SSE / df_n

    # Compute the F-statistic (session effect) - between observers/judges/methods (columns)
    SSC = ((np.mean(data, 0) - mean_data)**2).sum() * num_n
    MSC = SSC / df_m / num_n
    F_stat = MSC / MSE

    # Computer the subject effect - between measurements/observations/subjects (rows)
    SSR = sum_dist_mean - SSC - SSE
    MSR = SSR / df_n0

    # Compute variance between subjects
    var_obs = (MSR - MSE) / num_m

    # Computer ICC(3,1)
    ICC = (MSR - MSE) / (MSR + df_m * MSE)

    return ICC, df_m, df_n, F_stat, var_obs, MSE

# SETTINGS

#   Main Directory
DATADIR = "C:/Users/bdour/Documents/Work/Toronto/Sunnybrook/ACL Injury Screening/Data/"

#   Filenames
#       Note: for normalized data (varying between 0 and 1), edit filename below following this example: val_input_name = 'K_val_jc_input_X_norm.csv'
train_input_name = 'C_train_R_jc_input_X.csv'
train_target_name = 'V_train_RK_fl_target_y.csv'
train_real_name = 'C_train_RK_fl_real_y.csv'
val_input_name = 'C_val_R_jc_input_X.csv'
val_target_name = 'V_val_RK_fl_target_y.csv'
val_real_name = 'C_val_RK_fl_real_y.csv'
#       Trained model import
mod_name = 'seq2p_RK_fl_step1.h5'

#   Trial segmentation parameters
step_size = 1
seg_width = 20

#   Training parameters
batch_size = 32
#   Optimizer (choose between following options)
#       Adam: keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#       RMSprop: keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
opt = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

#   Number of random samples to plot
num_plot_sample = 5

# PATHS DEFINITION

train_input_path = os.path.join(DATADIR, 'Processed/Machine learning/Enhancement/Training data/Input data/' + train_input_name)
train_target_path = os.path.join(DATADIR, 'Processed/Machine learning/Enhancement/Training data/Target data/' + train_target_name)
train_real_path = os.path.join(DATADIR, 'Processed/Machine learning/Enhancement/Training data/Real data/' + train_real_name)
val_input_path = os.path.join(DATADIR, 'Processed/Machine learning/Enhancement/Validation data/Input data/' + val_input_name)
val_target_path = os.path.join(DATADIR, 'Processed/Machine learning/Enhancement/Validation data/Target data/' + val_target_name)
val_real_path = os.path.join(DATADIR, 'Processed/Machine learning/Enhancement/Validation data/Real data/' + val_real_name)
mod_path = os.path.join(DATADIR, 'Processed/Machine learning/Enhancement/' + mod_name)

# DATA & MODEL IMPORT

#   Import CSV files (X = Kinect joint centers trajectories - y = Vicon joint angles)
df_train_input_X = pd.read_csv(train_input_path)
df_train_target_y = pd.read_csv(train_target_path)
df_train_real_y = pd.read_csv(train_real_path)
df_val_input_X = pd.read_csv(val_input_path)
df_val_target_y = pd.read_csv(val_target_path)
df_val_real_y = pd.read_csv(val_real_path)

#   Import H5 file (LSTM RNN model)
model = tf.keras.models.load_model(mod_path)

print('')
print('-> Data & model imported')

#   Compile model
model.compile(optimizer=opt, loss='mae')

# DATA PROCESSING

#   Isolate numerical values in data
train_input_X, train_target_y, train_real_y = df_train_input_X.values, df_train_target_y.values, df_train_real_y.values
val_input_X, val_target_y, val_real_y = df_val_input_X.values, df_val_target_y.values, df_val_real_y.values

#   Reshape data into a (m, n, l) shaped array, where m represents the number of samples (= trials), n the length of each signal (changes across trials), and l the number of channels (= number of columns)

#       Initializations
train_input_Xr, train_target_yr, train_real_yr, train_idr = [], [], [], []
val_input_Xr, val_target_yr, val_real_yr, val_idr = [], [], [], []

#       Training data
for i in np.unique(train_input_X[:,1]):
    input_Xi = train_input_X[train_input_X[:,1] == i, 2:]
    target_yi = train_target_y[train_target_y[:,1] == i, 2]
    real_yi = train_real_y[train_real_y[:,1] == i, 2]
    idi = train_input_X[train_input_X[:,1] == i, 1]
    train_input_Xr.append(input_Xi)
    train_target_yr.append(target_yi)
    train_real_yr.append(real_yi)
    train_idr.append(idi)

#       Validation data
for i in np.unique(val_input_X[:,1]):
    input_Xi = val_input_X[val_input_X[:,1] == i, 2:]
    target_yi = val_target_y[val_target_y[:,1] == i, 2]
    real_yi = val_real_y[val_real_y[:,1] == i, 2]
    idi = val_input_X[val_input_X[:,1] == i, 1]
    val_input_Xr.append(input_Xi)
    val_target_yr.append(target_yi)
    val_real_yr.append(real_yi)
    val_idr.append(idi)

# CALCULATE AND PRINT RANGES

#   Input
range_input_X = []
for i in np.unique(train_input_X[:,1]):
    range_input_X.append(np.nanmax(train_input_X[train_input_X[:,1] == i, 2:], axis=0) - np.nanmin(train_input_X[train_input_X[:,1] == i, 2:], axis=0))
for i in np.unique(val_input_X[:,1]):
    range_input_X.append(np.nanmax(val_input_X[val_input_X[:,1] == i, 2:], axis=0) - np.nanmin(val_input_X[val_input_X[:,1] == i, 2:], axis=0))
range_input_X = np.array(range_input_X)     # convert nested list to numpy array
print('INPUT RANGES')
print(' Hip joint')
print('     Min range for input (training and validation): X -> ', np.round(np.nanmin(range_input_X[:,0]), 2), ' | Y -> ', np.round(np.nanmin(range_input_X[:,1]), 2), ' | Z -> ', np.round(np.nanmin(range_input_X[:,2]), 2))
print('     Max range for input (training and validation): X -> ', np.round(np.nanmax(range_input_X[:,0]), 2), ' | Y -> ', np.round(np.nanmax(range_input_X[:,1]), 2), ' | Z -> ', np.round(np.nanmax(range_input_X[:,2]), 2))
print('     Mean range for input (training and validation): X -> ', np.round(np.nanmean(range_input_X[:,0]), 2), '[SD ', np.round(np.nanstd(range_input_X[:,0]), 2), '] | Y -> ', np.round(np.nanmean(range_input_X[:,1]), 2), '[ SD ', np.round(np.nanstd(range_input_X[:,1]), 2), '] | Z -> ', np.round(np.nanmean(range_input_X[:,2]), 2), '[SD ', np.round(np.nanstd(range_input_X[:,2]), 2), ']')
print(' Knee joint')
print('     Min range for input (training and validation): X -> ', np.round(np.nanmin(range_input_X[:,3]), 2), ' | Y -> ', np.round(np.nanmin(range_input_X[:,4]), 2), ' | Z -> ', np.round(np.nanmin(range_input_X[:,5]), 2))
print('     Max range for input (training and validation): X -> ', np.round(np.nanmax(range_input_X[:,3]), 2), ' | Y -> ', np.round(np.nanmax(range_input_X[:,4]), 2), ' | Z -> ', np.round(np.nanmax(range_input_X[:,5]), 2))
print('     Mean range for input (training and validation): X -> ', np.round(np.nanmean(range_input_X[:,3]), 2), '[SD ', np.round(np.nanstd(range_input_X[:,3]), 2), '] | Y -> ', np.round(np.nanmean(range_input_X[:,4]), 2), '[ SD ', np.round(np.nanstd(range_input_X[:,4]), 2), '] | Z -> ', np.round(np.nanmean(range_input_X[:,5]), 2), '[SD ', np.round(np.nanstd(range_input_X[:,5]), 2), ']')
print(' Ankle joint')
print('     Min range for input (training and validation): X -> ', np.round(np.nanmin(range_input_X[:,6]), 2), ' | Y -> ', np.round(np.nanmin(range_input_X[:,7]), 2), ' | Z -> ', np.round(np.nanmin(range_input_X[:,8]), 2))
print('     Max range for input (training and validation): X -> ', np.round(np.nanmax(range_input_X[:,6]), 2), ' | Y -> ', np.round(np.nanmax(range_input_X[:,7]), 2), ' | Z -> ', np.round(np.nanmax(range_input_X[:,8]), 2))
print('     Mean range for input (training and validation): X -> ', np.round(np.nanmean(range_input_X[:,6]), 2), '[SD ', np.round(np.nanstd(range_input_X[:,6]), 2), '] | Y -> ', np.round(np.nanmean(range_input_X[:,7]), 2), '[ SD ', np.round(np.nanstd(range_input_X[:,7]), 2), '] | Z -> ', np.round(np.nanmean(range_input_X[:,8]), 2), '[SD ', np.round(np.nanstd(range_input_X[:,8]), 2), ']')

#   Target
range_target_y = []
for i in np.unique(train_target_y[:,1]):
    range_target_y.append(np.nanmax(train_target_y[train_target_y[:,1] == i, 2]) - np.nanmin(train_target_y[train_target_y[:,1] == i, 2]))
for i in np.unique(val_target_y[:,1]):
    range_target_y.append(np.nanmax(val_target_y[val_target_y[:,1] == i, 2]) - np.nanmin(val_target_y[val_target_y[:,1] == i, 2]))

print('TARGET RANGES')
print('     Min range for target (training and validation): ', np.round(np.nanmin(range_target_y), 2))
print('     Max range for target (training and validation): ', np.round(np.nanmax(range_target_y), 2))
print('     Mean range for target (training and validation): ', np.round(np.nanmean(range_target_y), 2), '[SD ', np.round(np.nanstd(range_target_y), 2), ']')

#   Real
range_real_y = []
for i in np.unique(train_real_y[:,1]):
    range_real_y.append(np.nanmax(train_real_y[train_real_y[:,1] == i, 2]) - np.nanmin(train_real_y[train_real_y[:,1] == i, 2]))
for i in np.unique(val_real_y[:,1]):
    range_real_y.append(np.nanmax(val_real_y[val_real_y[:,1] == i, 2]) - np.nanmin(val_real_y[val_real_y[:,1] == i, 2]))

print('REAL RANGES')
print('     Min range for real (training and validation): ', np.round(np.nanmin(range_real_y), 2))
print('     Max range for real (training and validation): ', np.round(np.nanmax(range_real_y), 2))
print('     Mean range for real (training and validation): ', np.round(np.nanmean(range_real_y), 2), '[SD ', np.round(np.nanstd(range_real_y), 2), ']')

#   Convert reshaped lists into arrays
train_input_Xr, train_target_yr, train_real_yr, train_idr = np.array(train_input_Xr), np.array(train_target_yr), np.array(train_real_yr), np.array(train_idr)
val_input_Xr, val_target_yr, val_real_yr, val_idr = np.array(val_input_Xr), np.array(val_target_yr), np.array(val_real_yr), np.array(val_idr)

#   Segment each trial into multiple segments, with only one target value per segment (= last value of y for the corresponding segment)
seg = seg.transform.SegmentXY(step=step_size, width=seg_width, order='C', y_func=seg.transform.last)

#       Training data
train_input_Xs, train_target_ys, _ = seg.transform(train_input_Xr, train_target_yr)
_ , train_real_ys, _ = seg.transform(train_input_Xr, train_real_yr)
_ , train_ids, _ = seg.transform(train_input_Xr, train_idr)

#       Validation data
val_input_Xs, val_target_ys, _ = seg.transform(val_input_Xr, val_target_yr)
_ , val_real_ys, _ = seg.transform(val_input_Xr, val_real_yr)
_ , val_ids, _ = seg.transform(val_input_Xr, val_idr)

#   Identify when each trial starts and ends based on trial ID column
u_train_ids, train_idx = np.unique(train_ids, return_index=True)
u_val_ids, val_idx = np.unique(val_ids, return_index=True)

print('')
print('-> Data processed')

# COMPUTE PREDICTIONS

#   Apply model to make prediction
train_predic = model.predict(train_input_Xs, batch_size=batch_size)
train_predic = train_predic[:,0]
val_predic = model.predict(val_input_Xs, batch_size=batch_size)
val_predic = val_predic[:,0]

# ICC & CORRELATIONS

#   ICC (functional tests metrics)
#       Initializations
train_target_peak, train_real_peak, train_predic_peak = [], [], []
val_target_peak, val_real_peak, val_predic_peak = [], [], []
#       Training data
for i in train_idx[0:len(train_idx)]:
    target_peaki = np.max(train_target_ys[i:i+1])
    real_peaki = np.max(train_real_ys[i:i+1])
    predic_peaki = np.max(train_predic[i:i+1])
    train_target_peak.append(target_peaki)
    train_real_peak.append(real_peaki)
    train_predic_peak.append(predic_peaki)
train_ICC, _, _, _, _, _ = ICC_2way_mixed(np.transpose([train_target_peak, train_real_peak]))
train_ICC_ml, _, _, _, _, _ = ICC_2way_mixed(np.transpose([train_target_peak, train_predic_peak]))
#       Validation data
for i in val_idx[0:len(val_idx)]:
    target_peaki = np.max(val_target_ys[i:i+1])
    real_peaki = np.max(val_real_ys[i:i+1])
    predic_peaki = np.max(val_predic[i:i+1])
    val_target_peak.append(target_peaki)
    val_real_peak.append(real_peaki)
    val_predic_peak.append(predic_peaki)
val_ICC, _, _, _, _, _ = ICC_2way_mixed(np.transpose([val_target_peak, val_real_peak]))
val_ICC_ml, _, _, _, _, _ = ICC_2way_mixed(np.transpose([val_target_peak, val_predic_peak]))

print('')
print('-> FUNCTIONAL TEST METRICS RESULTS')
print('     TRAINING data:')
print('         ... between Vicon and Kinect:                 ICC = ', np.round(train_ICC, 3))
print('         ... between Vicon and ML-powered Kinect:      ICC = ', np.round(train_ICC_ml, 3))
print('     VALIDATION data:')
print('         ... between Vicon and Kinect:                 ICC = ', np.round(val_ICC, 3))
print('         ... between Vicon and ML-powered Kinect:      ICC = ', np.round(val_ICC_ml, 3))

#   R2 calculation (time series)
#       Training data
_, _, R, _, _ = linregress(train_target_ys, train_real_ys)
train_R2 = R**2
_, _, R, _, _ = linregress(train_target_ys, train_predic)
train_R2_ml = R**2
#       Validation data
_, _, R, _, _ = linregress(val_target_ys, val_real_ys)
val_R2 = R**2
_, _, R, _, _ = linregress(val_target_ys, val_predic)
val_R2_ml = R**2

print('')
print('-> TIME SERIES RESULTS')
print('     TRAINING data:')
print('         ... between Vicon and Kinect:                 R-squared = ', np.round(train_R2*100, 2), '%')
print('         ... between Vicon and ML-powered Kinect:      R-squared = ', np.round(train_R2_ml*100, 2), '%')
print('     VALIDATION data:')
print('         ... between Vicon and Kinect:                 R-squared = ', np.round(val_R2*100, 2), '%')
print('         ... between Vicon and ML-powered Kinect:      R-squared = ', np.round(val_R2_ml*100, 2), '%')

# PLOT PREDICTIONS

#   All validation data
plt.figure(figsize=(6, 4))
plt.plot(val_target_ys, c='royalblue', label='Target (Vicon)')
plt.plot(val_real_ys, c='chocolate', label='Real (Kinect)')
plt.plot(val_predic, c='chocolate', ls='--', label='Predicted (ML-powered Kinect)')
plt.ylabel('Right knee flexion')
plt.xlabel('Frame')
plt.title('Target vs. Real and Predicted')
plt.legend()
plt.show()

#       Select random trial index
rand_idx = int(np.random.randint(len(train_idx)-1, size=1))
#       Generate subplot
plt.subplot(321)
plt.plot(train_target_ys[train_idx[rand_idx]:train_idx[rand_idx+1]], c='royalblue', label='Target (Vicon)')
plt.plot(train_real_ys[train_idx[rand_idx]:train_idx[rand_idx+1]], c='chocolate', label='Real (Kinect)')
plt.plot(train_predic[train_idx[rand_idx]:train_idx[rand_idx+1]], c='chocolate', ls='--', label='Predicted (ML-powered Kinect)')
if train_target_name == 'V_train_RK_ab_target_y.csv' or train_target_name == 'V_train_LK_ab_target_y.csv':
    plt.ylabel('Knee abduction angle')
elif train_target_name == 'V_train_RK_fl_target_y.csv' or train_target_name == 'V_train_LK_fl_target_y.csv':
    plt.ylabel('Knee flexion angle')
#       Select random trial index
rand_idx = int(np.random.randint(len(train_idx)-1, size=1))
#       Generate subplot
plt.subplot(322)
plt.plot(train_target_ys[train_idx[rand_idx]:train_idx[rand_idx+1]], c='royalblue', label='Target (Vicon)')
plt.plot(train_real_ys[train_idx[rand_idx]:train_idx[rand_idx+1]], c='chocolate', label='Real (Kinect)')
plt.plot(train_predic[train_idx[rand_idx]:train_idx[rand_idx+1]], c='chocolate', ls='--', label='Predicted (ML-powered Kinect)')
if train_target_name == 'V_train_RK_ab_target_y.csv' or train_target_name == 'V_train_LK_ab_target_y.csv':
    plt.ylabel('Knee abduction angle')
elif train_target_name == 'V_train_RK_fl_target_y.csv' or train_target_name == 'V_train_LK_fl_target_y.csv':
    plt.ylabel('Knee flexion angle')
#       Select random trial index
rand_idx = int(np.random.randint(len(train_idx)-1, size=1))
#       Generate subplot
plt.subplot(323)
plt.plot(train_target_ys[train_idx[rand_idx]:train_idx[rand_idx+1]], c='royalblue', label='Target (Vicon)')
plt.plot(train_real_ys[train_idx[rand_idx]:train_idx[rand_idx+1]], c='chocolate', label='Real (Kinect)')
plt.plot(train_predic[train_idx[rand_idx]:train_idx[rand_idx+1]], c='chocolate', ls='--', label='Predicted (ML-powered Kinect)')
if train_target_name == 'V_train_RK_ab_target_y.csv' or train_target_name == 'V_train_LK_ab_target_y.csv':
    plt.ylabel('Knee abduction angle')
elif train_target_name == 'V_train_RK_fl_target_y.csv' or train_target_name == 'V_train_LK_fl_target_y.csv':
    plt.ylabel('Knee flexion angle')
#       Select random trial index
rand_idx = int(np.random.randint(len(train_idx)-1, size=1))
#       Generate subplot
plt.subplot(324)
plt.plot(train_target_ys[train_idx[rand_idx]:train_idx[rand_idx+1]], c='royalblue', label='Target (Vicon)')
plt.plot(train_real_ys[train_idx[rand_idx]:train_idx[rand_idx+1]], c='chocolate', label='Real (Kinect)')
plt.plot(train_predic[train_idx[rand_idx]:train_idx[rand_idx+1]], c='chocolate', ls='--', label='Predicted (ML-powered Kinect)')
if train_target_name == 'V_train_RK_ab_target_y.csv' or train_target_name == 'V_train_LK_ab_target_y.csv':
    plt.ylabel('Knee abduction angle')
elif train_target_name == 'V_train_RK_fl_target_y.csv' or train_target_name == 'V_train_LK_fl_target_y.csv':
    plt.ylabel('Knee flexion angle')
#       Select random trial index
rand_idx = int(np.random.randint(len(train_idx)-1, size=1))
#       Generate subplot
plt.subplot(325)
plt.plot(train_target_ys[train_idx[rand_idx]:train_idx[rand_idx+1]], c='royalblue', label='Target (Vicon)')
plt.plot(train_real_ys[train_idx[rand_idx]:train_idx[rand_idx+1]], c='chocolate', label='Real (Kinect)')
plt.plot(train_predic[train_idx[rand_idx]:train_idx[rand_idx+1]], c='chocolate', ls='--', label='Predicted (ML-powered Kinect)')
plt.xlabel('Frame')
if train_target_name == 'V_train_RK_ab_target_y.csv' or train_target_name == 'V_train_LK_ab_target_y.csv':
    plt.ylabel('Knee abduction angle')
elif train_target_name == 'V_train_RK_fl_target_y.csv' or train_target_name == 'V_train_LK_fl_target_y.csv':
    plt.ylabel('Knee flexion angle')
#       Select random trial index
rand_idx = int(np.random.randint(len(train_idx)-1, size=1))
#       Generate subplot
plt.subplot(326)
plt.plot(train_target_ys[train_idx[rand_idx]:train_idx[rand_idx+1]], c='royalblue', label='Target (Vicon)')
plt.plot(train_real_ys[train_idx[rand_idx]:train_idx[rand_idx+1]], c='chocolate', label='Real (Kinect)')
plt.plot(train_predic[train_idx[rand_idx]:train_idx[rand_idx+1]], c='chocolate', ls='--', label='Predicted (ML-powered Kinect)')
plt.xlabel('Frame')
if train_target_name == 'V_train_RK_ab_target_y.csv' or train_target_name == 'V_train_LK_ab_target_y.csv':
    plt.ylabel('Knee abduction angle')
elif train_target_name == 'V_train_RK_fl_target_y.csv' or train_target_name == 'V_train_LK_fl_target_y.csv':
    plt.ylabel('Knee flexion angle')

plt.suptitle('TRAINING DATA\nTarget vs. Real and Predicted')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

#   Random VALIDATION samples

#       Select random trial index
rand_idx = int(np.random.randint(len(val_idx)-1, size=1))
#       Generate subplot
plt.subplot(321)
plt.plot(val_target_ys[val_idx[rand_idx]:val_idx[rand_idx+1]], c='royalblue', label='Target (Vicon)')
plt.plot(val_real_ys[val_idx[rand_idx]:val_idx[rand_idx+1]], c='chocolate', label='Real (Kinect)')
plt.plot(val_predic[val_idx[rand_idx]:val_idx[rand_idx+1]], c='chocolate', ls='--', label='Predicted (ML-powered Kinect)')
if train_target_name == 'V_train_RK_ab_target_y.csv' or train_target_name == 'V_train_LK_ab_target_y.csv':
    plt.ylabel('Knee abduction angle')
elif train_target_name == 'V_train_RK_fl_target_y.csv' or train_target_name == 'V_train_LK_fl_target_y.csv':
    plt.ylabel('Knee flexion angle')
#       Select random trial index
rand_idx = int(np.random.randint(len(val_idx)-1, size=1))
#       Generate subplot
plt.subplot(322)
plt.plot(val_target_ys[val_idx[rand_idx]:val_idx[rand_idx+1]], c='royalblue', label='Target (Vicon)')
plt.plot(val_real_ys[val_idx[rand_idx]:val_idx[rand_idx+1]], c='chocolate', label='Real (Kinect)')
plt.plot(val_predic[val_idx[rand_idx]:val_idx[rand_idx+1]], c='chocolate', ls='--', label='Predicted (ML-powered Kinect)')
if train_target_name == 'V_train_RK_ab_target_y.csv' or train_target_name == 'V_train_LK_ab_target_y.csv':
    plt.ylabel('Knee abduction angle')
elif train_target_name == 'V_train_RK_fl_target_y.csv' or train_target_name == 'V_train_LK_fl_target_y.csv':
    plt.ylabel('Knee flexion angle')
#       Select random trial index
rand_idx = int(np.random.randint(len(val_idx)-1, size=1))
#       Generate subplot
plt.subplot(323)
plt.plot(val_target_ys[val_idx[rand_idx]:val_idx[rand_idx+1]], c='royalblue', label='Target (Vicon)')
plt.plot(val_real_ys[val_idx[rand_idx]:val_idx[rand_idx+1]], c='chocolate', label='Real (Kinect)')
plt.plot(val_predic[val_idx[rand_idx]:val_idx[rand_idx+1]], c='chocolate', ls='--', label='Predicted (ML-powered Kinect)')
if train_target_name == 'V_train_RK_ab_target_y.csv' or train_target_name == 'V_train_LK_ab_target_y.csv':
    plt.ylabel('Knee abduction angle')
elif train_target_name == 'V_train_RK_fl_target_y.csv' or train_target_name == 'V_train_LK_fl_target_y.csv':
    plt.ylabel('Knee flexion angle')
#       Select random trial index
rand_idx = int(np.random.randint(len(val_idx)-1, size=1))
#       Generate subplot
plt.subplot(324)
plt.plot(val_target_ys[val_idx[rand_idx]:val_idx[rand_idx+1]], c='royalblue', label='Target (Vicon)')
plt.plot(val_real_ys[val_idx[rand_idx]:val_idx[rand_idx+1]], c='chocolate', label='Real (Kinect)')
plt.plot(val_predic[val_idx[rand_idx]:val_idx[rand_idx+1]], c='chocolate', ls='--', label='Predicted (ML-powered Kinect)')
if train_target_name == 'V_train_RK_ab_target_y.csv' or train_target_name == 'V_train_LK_ab_target_y.csv':
    plt.ylabel('Knee abduction angle')
elif train_target_name == 'V_train_RK_fl_target_y.csv' or train_target_name == 'V_train_LK_fl_target_y.csv':
    plt.ylabel('Knee flexion angle')
#       Select random trial index
rand_idx = int(np.random.randint(len(val_idx)-1, size=1))
#       Generate subplot
plt.subplot(325)
plt.plot(val_target_ys[val_idx[rand_idx]:val_idx[rand_idx+1]], c='royalblue', label='Target (Vicon)')
plt.plot(val_real_ys[val_idx[rand_idx]:val_idx[rand_idx+1]], c='chocolate', label='Real (Kinect)')
plt.plot(val_predic[val_idx[rand_idx]:val_idx[rand_idx+1]], c='chocolate', ls='--', label='Predicted (ML-powered Kinect)')
plt.xlabel('Frame')
if train_target_name == 'V_train_RK_ab_target_y.csv' or train_target_name == 'V_train_LK_ab_target_y.csv':
    plt.ylabel('Knee abduction angle')
elif train_target_name == 'V_train_RK_fl_target_y.csv' or train_target_name == 'V_train_LK_fl_target_y.csv':
    plt.ylabel('Knee flexion angle')
#       Select random trial index
rand_idx = int(np.random.randint(len(val_idx)-1, size=1))
#       Generate subplot
plt.subplot(326)
plt.plot(val_target_ys[val_idx[rand_idx]:val_idx[rand_idx+1]], c='royalblue', label='Target (Vicon)')
plt.plot(val_real_ys[val_idx[rand_idx]:val_idx[rand_idx+1]], c='chocolate', label='Real (Kinect)')
plt.plot(val_predic[val_idx[rand_idx]:val_idx[rand_idx+1]], c='chocolate', ls='--', label='Predicted (ML-powered Kinect)')
plt.xlabel('Frame')
if train_target_name == 'V_train_RK_ab_target_y.csv' or train_target_name == 'V_train_LK_ab_target_y.csv':
    plt.ylabel('Knee abduction angle')
elif train_target_name == 'V_train_RK_fl_target_y.csv' or train_target_name == 'V_train_LK_fl_target_y.csv':
    plt.ylabel('Knee flexion angle')

plt.suptitle('VALIDATION DATA\nTarget vs. Real and Predicted')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

print('')
print('--------------')
print('CODE COMPLETED')
print('--------------')
