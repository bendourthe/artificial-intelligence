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

#   Epochs
epoch_num = 50

#   Accuracy threshold: will stop training when reached
acc_lim = 99.9    # in %

# DEFINE CALLBACK

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>acc_lim/100):
      print("\nTraining stopped -> accuracy threshold of " + str(acc_lim) + '% reached')
      self.model.stop_training = True

print('__')
print('Callback defined')

# DIRECTORIES

#   Define main directories
train_dir = '/tmp/training'
validation_dir = '/tmp/validation'

#   Identify training and validation directories
train_horses_dir = os.path.join(train_dir, 'horses')
train_humans_dir = os.path.join(train_dir, 'humans')
validation_horses_dir = os.path.join(validation_dir, 'horses')
validation_humans_dir = os.path.join(validation_dir, 'humans')

#   Identify filenames
train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print('__')
print('Directories defined')

# DATA IMPORT

#   Get the Horse or Human training dataset
!wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip -O /tmp/horse-or-human.zip

#   Get the Horse or Human validation dataset
!wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip -O /tmp/validation-horse-or-human.zip

#   Unzip imported data
#       Training
local_zip = '//tmp/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/training')
zip_ref.close()
#       Validation
local_zip = '//tmp/validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation')
zip_ref.close()

print('__')
print('Data import completed')

# TRANSFER LEARNING

#   Download the inception v3 weights
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

#   Import the inception model
from tensorflow.keras.applications.inception_v3 import InceptionV3

#   Create an instance of the inception model from the local pre-trained weights
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150, 150, 3),
                                include_top = False,
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

#   Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
  layer.trainable = False

#   Define last layer of pre-trained model to be used as input layer of the new local CNN
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

print('__')
print('Transfer learning completed')

# LOCAL CONVOLUTIONAL NEURAL NETWORK DEFINITION

#   Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
#   Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
#   Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
#   Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)
#   Save model
model = Model(pre_trained_model.input, x)

#   Compile model
model.compile(optimizer = RMSprop(lr=0.0001),
              loss = 'binary_crossentropy',
              metrics = ['acc'])

print('__')
print('Model defined and compiled')

# DATA AUGMENTATION

#   Add data augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

#   Rescale validation data set (note: should not be augmented!)
test_datagen = ImageDataGenerator(rescale = 1./255.)

#   Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20,
                                                    class_mode = 'binary',
                                                    target_size = (150, 150))

#   Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size  = 20,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))

print('__')
print('Data augmentation completed')

# MODEL TRAINING

callbacks = myCallback()
history = model.fit_generator(train_generator,
                              validation_data = validation_generator,
                              steps_per_epoch = 100,
                              epochs = 200,
                              validation_steps = 50,
                              verbose = 2,
                              callbacks=[callbacks])

print('__')
print('Model trained')

# PLOT TRAINING

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()

print('__')
print('Plot generated')

print('--------------')
print('CODE COMPLETED')
print('--------------')
