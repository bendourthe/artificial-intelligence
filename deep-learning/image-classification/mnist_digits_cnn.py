# LIBRARIES IMPORT

import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras

# SETTINGS

cb = 1              # Do you want to use callbacks? (1: Yes - 0: No)   
acc_lim = 99.8      # Accuracy threshold (in %) - will stop training the neural network once this level is reached
num_epochs = 20     # Number of epochs

# DEFINE CALLBACK

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > acc_lim / 100):
            print('\n\nTraining stopped after reaching ' + str(acc_lim) + '% accuracy')
            self.model.stop_training = True
      
callbacks = myCallback()

# IMPORT AND LOAD MNIST DIGITS DATA

mnist = keras.datasets.mnist
(training_set, training_labels), (test_set, test_labels) = mnist.load_data()

print('STEP 1: Data imported')

# DATA PROCESSING

#    Reshape data
#        Instead of having N images of size (M x M), we need to reshape the data set
#        to have all the images in a 4D matrix of size (N x M x M x B) (with B being the number of bits)
training_set = training_set.reshape(60000, 28, 28, 1)
test_set = test_set.reshape(10000, 28, 28, 1)

#    Normalize data
#        Original data a represented by grayscale values between 0 and 255.
#        However, it's easier for the neural network to deal with values between 0 and 1.
training_set = training_set/255
test_set = test_set/255

print('STEP 2: Data processing')

# DEFINE NEURAL NETWORK

model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation = 'relu'), \
    keras.layers.Dense(10, activation = 'softmax')]) 

print('STEP 3: Neural network defined')

# COMPILE NEURAL NETWORK

model.compile(optimizer = 'adam', \
              loss = 'sparse_categorical_crossentropy', \
              metrics = ['accuracy'])

print('STEP 4: Neural network compiled')

# TRAIN MODEL

if cb == 1:
    model.fit(training_set, training_labels, epochs = num_epochs, verbose=2, callbacks=[myCallback()])
else:
    model.fit(training_set, training_labels, epochs = num_epochs, verbose=2)

print('STEP 5: Neural network trained')

# EVALUATE MODEL ON TEST SET

nn_eval = model.evaluate(test_set, test_labels, verbose=0)

print('STEP 6: Neural network evaluated on test images with accuracy of ' + str(np.round(nn_eval[1],4)*100) + '%')

# MODEL PREDICTIONS

n = random.randint(0,len(test_set)+1)

classification = model.predict(test_set)
print('STEP 7: Neural network predictions generated')
print('    Random test:')
print('        Original digit label ->                  ' + str(test_labels[n]))
print('        Digit predicted by neural network ->     ' + str(np.argmax(classification[n])))
