# LIBRARIES IMPORT

import os
import random

# FUNCTION

def split_data(source_dir, train_dir, test_dir, split_ratio):
    '''
    Shuffles files within a source directory and redistribute them in a training and testing subdirectories.
    Note: each file is checked and passed in case the corresponding file length is zero (empty file)
    Input:
        source_dir: source directory containing files
        train_dir: training directory where a portion of the files will be copied
        test_dir: testing directory where the remaining files will be copied
        split_ratio: percentage of the files to be copied into the training directory (e.g. split_ratio = 90, for 90%)
    '''
    # Get list of files contained in the source directory
    files = os.listdir(source_dir)
    # Shuffle files
    shuffled_files = random.sample(files, len(files))
    split_ratio = split_ratio/100
    # Calculate length of the training and testing sets
    train_length = int(split_ratio*len(files))
    test_length = int(len(files) - train_length)
    # Generate training and testing sets
    train_files = shuffled_files[0:train_length]
    test_files = shuffled_files[-test_length:]
    # Copy training files to the training subdirectory
    for file in train_files:
        if os.path.getsize(source_dir + file) == 0:
            print(file + ' is zero length, so ignoring')
        else:
            copyfile(source_dir + file, train_dir + file)
    # Copy testing files to the training subdirectory
    for file in test_files:
        if os.path.getsize(source_dir + file) == 0:
            print(file + ' is zero length, so ignoring')
        else:
            copyfile(source_dir + file, test_dir + file)
