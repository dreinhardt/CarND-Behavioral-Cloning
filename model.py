import csv
import cv2
import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, merge, Input, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv2D
import tensorflow as tf
from keras import backend as K


print('\n \n')


###################################################
############# Configuration #######################
###################################################
#os.chdir('Project-4/CarND-Behavioral-Cloning-P3')
basePath = ''
base_data_path = basePath + './data/'
batch_size= 32      # Set our batch size
keep_prob = 0.5     # 50% Dropchance
num_epochs = 3      # 5 complete rounds

###################################################
############# Start of Functions ##################
##################################################c
def loadData():         # get the images and separte them into set(X) and labels(y)
    lines = []
    with open(base_data_path + 'driving_log.csv') as cvsfile:
        reader = csv.reader(cvsfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    
    lines.pop(0) # remember to delete the first line in driving_log

    for line in lines:
        current_path = base_data_path + line[0]
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

    X_train = np.array(images)
    y_train = np.array(measurements)

    return X_train, y_train


def loadSamples():         # get the images
    samples = []
    with open(base_data_path + 'driving_log.csv') as cvsfile:
        reader = csv.reader(cvsfile)
        for line in reader:
            samples.append(line)
    return samples


def generator(samples, batch_size):      # load data and preprocess it on the fly, in batch size portions to feed into your Behavioral Cloning model
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)        # mix it :-D
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for n in range(3):  # all 3 angles are computed as a center image
                    name = base_data_path + 'IMG/' + batch_sample[n].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3])

                    if abs(angle) > 0:
                        if n == 1:          # Image left side
                            angle = angle + 0.2  # correct like in tutorial
                        if n == 2:          # Image right side
                            angle = angle - 0.2  # correct like in tutorial
                        images.append(image)
                        image_flipped = np.fliplr(image)
                        images.append(image_flipped)     # flip the image like within the tutorial
                        angles.append(angle)
                        angles.append(angle*(-1))        # invert the angle
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


###################################################
############# Setup the model #####################
###################################################
model = Sequential()

# Imitating the network of NVidia :-D

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# Crop image to only see section with road
model.add(Cropping2D(cropping=((70,25),(0,0))))     # sizes from tutorial video
# Convolution 5x5 Layers
model.add(Conv2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(48,5,5,subsample=(2,2),activation='relu'))
# Convolution 3x3 Layers
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
model.add(Dropout(keep_prob))
model.add(Flatten())
# Full-Connected Layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')


###################################################
############# Get Data and Start Computing ########
###################################################

# compile and train the model using the generator function
# X_train, y_train = loadData()     # old function to load data without generator
samples = loadSamples()
train_samples, validation_samples = train_test_split(samples, test_size=0.2)        # used from example in tutorial, to split a small validation portion
train_generator = generator(train_samples, batch_size=batch_size)                   # training the model
validation_generator = generator(validation_samples, batch_size=batch_size)         # validate the model


# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=num_epochs)  # make the number of epochs adjustable, old function without generator
model.fit_generator(generator=train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, validation_steps=np.ceil(len(validation_samples)/batch_size), epochs=num_epochs, verbose=1)  # make the number of epochs adjustable, verbose shows the output


###################################################
############# END #################################
###################################################
model.save('model.h5')  # save the model
print('\n \n')