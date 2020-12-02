# LIBRARIES IMPORT

import os
import zipfile
import random
import tensorflow as tf
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from google.colab import files
from keras.preprocessing import image

# SETTINGS

#   Percentage of the data to be set as training data
split_size = 90

#   Epochs
epoch_num = 50

# DEPENDENCIES

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

# DATA IMPORT

#   Download full Cats-v-Dogs dataset
!wget --no-check-certificate \
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" -O "/tmp/cats-and-dogs.zip"
#   Unzip downloaded file to /tmp and create a tmp/PetImages directory containing subdirectories called 'Cat' and 'Dog'
local_zip = '/tmp/cats-and-dogs.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

#   Note: If the URL doesn't work, go to: https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
#   and right click on the 'Download Manually' link to get a new URL

print('__')
print('Data import completed')

# DATA CHECK

print('__')
print('Number of elements in the Cat subdirectory: ' + str(len(os.listdir('/tmp/PetImages/Cat/'))))
print('Number of elements in the Dog subdirectory: ' + str(len(os.listdir('/tmp/PetImages/Dog/'))))

# CREATE NEW SUBDIRECTORIES

try:
    os.mkdir('/tmp/cats-v-dogs')
    os.mkdir('/tmp/cats-v-dogs/training')
    os.mkdir('/tmp/cats-v-dogs/testing')
    os.mkdir('/tmp/cats-v-dogs/training/cats')
    os.mkdir('/tmp/cats-v-dogs/training/dogs')
    os.mkdir('/tmp/cats-v-dogs/testing/cats')
    os.mkdir('/tmp/cats-v-dogs/testing/dogs')
    print('__')
    print('Training and Testing subdirectories created')
except OSError:
    print('__')
    print('Training and Testing subdirectories NOT created')
    pass

# DATA SPLIT

#   Identify source, training and testing directories
cat_source_dir = "/tmp/PetImages/Cat/"
cat_train_dir = "/tmp/cats-v-dogs/training/cats/"
cat_test_dir = "/tmp/cats-v-dogs/testing/cats/"
dog_source_dir = "/tmp/PetImages/Dog/"
dog_train_dir = "/tmp/cats-v-dogs/training/dogs/"
dog_test_dir = "/tmp/cats-v-dogs/testing/dogs/"

#   Copy files from source to training and testing subdirectories
split_data(cat_source_dir, cat_train_dir, cat_test_dir, split_size)
split_data(dog_source_dir, dog_train_dir, dog_test_dir, split_size)

#   Check number of files in each subdirectory
print('Number of training files (Cats): ' + str(len(os.listdir('/tmp/cats-v-dogs/training/cats/'))))
print('Number of training files (Dogs): ' + str(len(os.listdir('/tmp/cats-v-dogs/training/dogs/'))))
print('Number of testing files (Cats): ' + str(len(os.listdir('/tmp/cats-v-dogs/testing/cats/'))))
print('Number of testing files (Dogs): ' + str(len(os.listdir('/tmp/cats-v-dogs/testing/dogs/'))))

print('__')
print('Data distribution completed')

# DEEP CONVOLUTIONAL NEURAL NETWORK DEFINITION

#   Define model
model = tf.keras.models.Sequential([ \
    # Convolution layers
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)), \
    tf.keras.layers.MaxPooling2D(2,2), \
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'), \
    tf.keras.layers.MaxPooling2D(2,2), \
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), \
    tf.keras.layers.MaxPooling2D(2,2), \
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), \
    # Add a hidden layer of 512 neurons
    tf.keras.layers.Dense(512, activation='relu'), \
    # Single neuron output layer (value between 0 and 1 for binary classification)
    tf.keras.layers.Dense(1, activation='sigmoid')  \
])

#   Compile model
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

print('__')
print('DCNN defined and compiled')

# DATA AUGMENTATION

#   Training data
train_dir = '/tmp/cats-v-dogs/training'
train_datagen = ImageDataGenerator( \
    rescale=1./255, \
    rotation_range=40, \
    width_shift_range=0.2, \
    height_shift_range=0.2, \
    shear_range=0.2, \
    zoom_range=0.2, \
    horizontal_flip=True, \
    fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(train_dir, \
                                                    batch_size=128, \
                                                    class_mode='binary', \
                                                    target_size=(150, 150))

#   Validation data
val_dir = '/tmp/cats-v-dogs/testing'
validation_datagen = ImageDataGenerator( \
    rescale=1./255, \
    rotation_range=40, \
    width_shift_range=0.2, \
    height_shift_range=0.2, \
    shear_range=0.2, \
    zoom_range=0.2, \
    horizontal_flip=True, \
    fill_mode='nearest')
validation_generator = validation_datagen.flow_from_directory(val_dir, \
                                                    batch_size=32, \
                                                    class_mode='binary', \
                                                    target_size=(150, 150))

print('__')
print('Data augmentation completed')

# MODEL TRAINING

history = model.fit_generator(train_generator, \
                              epochs=epoch_num, \
                              verbose=1, \
                              validation_data=validation_generator)

print('__')
print('Model trained')

# LOSS AND ACCURACY VISUALIZATION

%matplotlib inline

#   Retrieve results on training and test data for each training epoch
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

#   Get number of epochs
epochs=range(len(acc))

#   Plot training and validation accuracy per epoch
plt.plot(epochs, acc, "Training Accuracy")
plt.plot(epochs, val_acc, "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss, "Training Loss")
plt.plot(epochs, val_loss, "Validation Loss")
plt.title('Training and validation loss')

# UPLOAD NEW IMAGE AND MAKE PREDICTION

#   Upload image from folder (manual selection)
uploaded = files.upload()

for fn in uploaded.keys():
    # Import image and convert
    path = '/content/' + fn
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    # Apply model to make prediction
    classes = model.predict(images, batch_size=10)
    # Display prediction
    print(classes[0])
    if classes[0]>0.5:
        print(fn + " is a dog")
    else:
        print(fn + " is a cat")

print('__')
print('Plot generated')

print('--------------')
print('CODE COMPLETED')
print('--------------')
