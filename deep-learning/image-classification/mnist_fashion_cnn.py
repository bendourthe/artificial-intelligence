# LIBRARIES IMPORT

import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras.layers.core import Activation

# SETTINGS

# Define desired level of classification accuracy on the training data (once reached, will stop epochs)
desired_acc = 99

# DEFINE CALLBACK

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc') > desired_acc/100):
      print('\nReached ' + str(desired_acc) + '% accuracy -> training stopped!')
      self.model.stop_training = True

callbacks = myCallback()

# IMPORT AND LOAD MNIST DIGITS DATA

mnist = keras.datasets.fashion_mnist
(training_set, training_labels), (test_set, test_labels) = mnist.load_data()

print('STEP 1: Data imported')

# DATA PROCESSING

#    Data reshape
training_set = training_set.reshape(60000, 28, 28, 1)
test_set = test_set.reshape(10000, 28, 28, 1)

#    Data normalization
#        Original data a represented by grayscale values between 0 and 255.
#        However, it's easier for the neural network to deal with values between 0 and 1.

training_set = training_set/255
test_set = test_set/255

print('STEP 2: Data processed')

# DEFINE CONVOLUTIONAL NEURAL NETWORK

model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (28, 28, 1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')])

print('STEP 3: Convolutional neural network defined')

# COMPILE NEURAL NETWORK

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy', \
              metrics = ['accuracy'])
model.summary()

print('STEP 4: Convolutional neural network compiled')

# TRAIN MODEL

model.fit(training_set, training_labels, epochs = 50, verbose=1, callbacks=[myCallback()])

print('STEP 5: Neural network trained')

# EVALUATE MODEL ON TEST SET

nn_eval = model.evaluate(test_set, test_labels, verbose=0)

print('STEP 6: Neural network evaluated on test images with accuracy of ' + str(np.round(nn_eval[1],4)*100) + '%')

# MODEL PREDICTIONS

n = random.randint(0,len(test_set)+1)
classification = model.predict(test_set)
print('STEP 7: Neural network predictions generated')
print('    Random test:')
print('        Original item label ->                        ' + str(test_labels[n]))
print('        Item label predicted by neural network ->     ' + str(np.argmax(classification[n])))
