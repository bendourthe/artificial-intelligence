# LIBRARIES IMPORT

import os
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy import interpolate

# SETTINGS

#   Main Directory
DATADIR = "C:/Users/bdour/Documents/Work/Toronto/Sunnybrook/Projects/ACL Injury Screening/Data"

#   Data comparison
#       Choose between 'Vicon vs. Kinect' and 'Vicon vs. Curv'
data_comp = 'Vicon vs. Kinect'

#   List of participants
#       Full list -> ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26']
#       List of participants with complete data -> ['01', '02', '03', '04', '05', '06', '07', '08', '11', '12', '13', '15', '16', '18', '19', '20', '21', '22', '23', '24', '25', '26']
participants = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26']

#   List of trials
#       Full list -> ['DVJ_0', 'DVJ_1', 'DVJ_2', 'RDist_0', 'RDist_1', 'RDist_2', 'LDist_0', 'LDist_1', 'LDist_2', 'RTimed_0', 'RTimed_1', 'RTimed_2', 'LTimed_0', 'LTimed_1', 'LTimed_2']
trials = ['DVJ_0', 'DVJ_1', 'DVJ_2', 'RDist_0', 'RDist_1', 'RDist_2', 'LDist_0', 'LDist_1', 'LDist_2', 'RTimed_0', 'RTimed_1', 'RTimed_2', 'LTimed_0', 'LTimed_1', 'LTimed_2']

#   Extensions
ext_data = '.csv'

# INITIALIZATIONS

#       Generate empty DataFrames
df = pd.DataFrame()

for participant in participants:
    print('')
    print('Participant in progress -> ' + participant + '...')
    i = 1       # Used to create unique trial IDs

    for trial in trials:
        print('...')
        print('Trial in progress -> ' + trial)

# EXCEPTIONS

        #   Skip trials that are missing or that are corrupted
        if participant == '09' and (trial == 'DVJ_0' or trial == 'DVJ_1' or trial == 'DVJ_2' or trial == 'RDist_0' or trial == 'RDist_1' or trial == 'RDist_2') or participant == '10' and trial == 'LTimed_0' or participant == '17' and trial == 'LDist_0':

            print('-> Warning! Skipped due to missing/corrupted data')

        else:

# IMPORT PATHS DEFINITION

        #   Import
            #       Joint centers
            exec("jc_path = os.path.join(DATADIR, 'Processed/Comparison/" + data_comp + '/Resampled and cropped data/Joint centers/' + participant + '_' + trial + '_processed_jc' + ext_data + "')")
            #       Biomechanical variables
            exec("bvars_path = os.path.join(DATADIR, 'Processed/Comparison/" + data_comp + '/Resampled and cropped data/Biomechanical variables/IFT Kinect/' + participant + '_' + trial + '_processed_bvars' + ext_data + "')")

# DATA IMPORT

    #   Joint centers

        #       Import from csv and convert to numpy array
            jc = pd.read_csv(jc_path, skiprows=1)
            jc = np.array(jc)

        #       Create distinct arrays for Vicon and Kinect/Curv data
            jc_V = jc[:,2:20]                   # V for Vicon
            jc_KC = jc[:,22:]                   # KC for Kinect or Curv (depending on what's selected and to keep notation consistent)

    #   Biomechanical variables

        #       Import from csv and convert to numpy array
            bvars = pd.read_csv(bvars_path, skiprows=1)
            bvars = np.array(bvars)

        #           Create distinct arrays for Vicon and Kinect/Curv data
            bvars_V = bvars[:,2:9]                   # V for Vicon
            bvars_KC = bvars[:,11:]                  # KC for Kinect or Curv (depending on what's selected and to keep notation consistent)
            #       If Dist and Timed are included, then KASR will be ignored (to avoid presence of NaN in the data extracted from Dist and Timed trials, where KASR was not calculated)
            if trial == 'RDist_0' or trial == 'RDist_1' or trial == 'RDist_2' or trial == 'LDist_0' or trial == 'LDist_1' or trial == 'LDist_2' or trial == 'RTimed_0' or trial == 'RTimed_1' or trial == 'RTimed_2' or trial == 'LTimed_0' or trial == 'LTimed_1' or trial == 'LTimed_2':
                bvars_V[:,-1] = np.NaN
                bvars_KC[:,-1] = np.NaN

        #       Ensure jc and bvars have same length (crop bvars to length of jc or the opposite)
            if len(bvars_V) < len(jc_V):
                jc_V = jc_V[0:len(bvars_V),:]
                jc_KC = jc_KC[0:len(jc_V),:]
            elif len(bvars_V) > len(jc_V):
                if len(np.shape(bvars_V)) > 1:
                    bvars_V = bvars_V[0:len(jc_V),:]
                    bvars_KC = bvars_KC[0:len(bvars_V),:]
                else:
                    bvars_V = bvars_V[0:len(jc_V)]
                    bvars_KC = bvars_KC[0:len(bvars_V)]

            print('-> Trial data imported')

# DATA COMPILATION

        #   Participant label column
            df_part = pd.DataFrame({'participant': [participant]*len(jc_V)})

        #   Trial category
            df_cat = pd.DataFrame({'trial_cat': [trial.split('_')[0]]*len(jc_V)})

        #   Trial ID
            df_id = pd.DataFrame({'trial_id': [participant+'0'+str(i)]*len(jc_V)})

        #   Joint centers from Kinect/Curv
            df_jc_V = pd.DataFrame(jc_V, columns = ['V_RHip_x','V_RHip_y','V_RHip_z','V_RKnee_x','V_RKnee_y','V_RKnee_z','V_RAnkle_x','V_RAnkle_y','V_RAnkle_z','V_LHip_x','V_LHip_y','V_LHip_z','V_LKnee_x','V_LKnee_y','V_LKnee_z','V_LAnkle_x','V_LAnkle_y','V_LAnkle_z'])

        #   Joint centers from Kinect/Curv
            df_jc_KC = pd.DataFrame(jc_KC, columns = ['K_RHip_x','K_RHip_y','K_RHip_z','K_RKnee_x','K_RKnee_y','K_RKnee_z','K_RAnkle_x','K_RAnkle_y','K_RAnkle_z','K_LHip_x','K_LHip_y','K_LHip_z','K_LKnee_x','K_LKnee_y','K_LKnee_z','K_LAnkle_x','K_LAnkle_y','K_LAnkle_z'])

        #   Joint angles from Vicon (target)
            df_bvars_V = pd.DataFrame(bvars_V, columns = ['V_RHip_fl','V_RKnee_fl','V_RKnee_ab','V_LHip_fl','V_LKnee_fl','V_LKnee_ab','V_KASR'])

        #   Joint angles from Kinect/Curv (for comparison with model prediction)
            df_bvars_KC = pd.DataFrame(bvars_KC, columns = ['K_RHip_fl','K_RKnee_fl','K_RKnee_ab','K_LHip_fl','K_LKnee_fl','K_LKnee_ab','K_KASR'])

        #   Concatenate trial Input, Target and Real DataFrames to Participant and Trial label DataFrames
            df_trial = pd.concat([df_part, df_cat, df_id, df_jc_V, df_jc_KC, df_bvars_V, df_bvars_KC], axis=1)

        #   Concatenate trial data with total data
            df = pd.concat([df, df_trial], axis=0)

            print('-> Trial data compiled')

            i = i + 1

print('')
print('Data compiled')

# EXPORT PATHS DEFINITION

if data_comp == 'Vicon vs. Kinect':
    export_path = os.path.join(DATADIR, 'Processed/Machine learning/Enhancement/V-vs-K_data.csv')

elif data_comp == 'Vicon vs. Curv':
    export_path = os.path.join(DATADIR, 'Processed/Machine learning/Enhancement/V-vs-C_data.csv')

# DATA EXPORT

df.to_csv(export_path, index=False)

print('')
print('Data exported')


print('')
print('--------------')
print('CODE COMPLETED')
print('--------------')

