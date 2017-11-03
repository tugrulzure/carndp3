import os
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, ELU, Dropout

samples = []

with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
del(samples[0]) # Delete the header, or the generator will throw an error. Tip from the udacity forum.

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print("no of training samples",len(train_samples))
print("no of validation samples", len(validation_samples))

def generator(samples, batch_size=32): #A batch size of 32 will output 32*3 images and angles for center, left and right.
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            correction=0.25 #Correction factor for left and right images.
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = './IMG/'+batch_sample[0].split('/')[-1]
                left_name = './IMG/'+batch_sample[1].split('/')[-1]
                right_name = './IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(center_name)
                center_image = cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB) #It is important to convert BGR to RGB.
                left_image = cv2.imread(left_name)
                left_image = cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB)
                right_image = cv2.imread(right_name)
                right_image = cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                left_angle = center_angle+correction #Apply correction factors
                right_angle = center_angle-correction
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

def resize_and_normalize(X_train): #Resize and normalize the image / Tip from the udacity forums
    from keras.backend import tf as ktf
    resized = ktf.image.resize_images(X_train, (64,64)) #Resizing to 64x64 results in very short training time per epoch, around 100s.
    resized = resized / 255 - 0.5 #Normalize the images
    return resized

ch, row, col = 3, 160, 320  # Image format

#PilotNet model is from nVidia paper "Explaining How a Deep Neural Network is Trained with End-to-End Learning Steers a Car"
model = Sequential()
model.add(Cropping2D(cropping=((70,25),(1,1)), input_shape=(160,320,3)))
model.add(Lambda(resize_and_normalize))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="elu")) #ELU activations are said to perform better than Relu's.
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="elu")) #https://arxiv.org/abs/1511.07289
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="elu"))
model.add(Convolution2D(64,3,3,activation="elu"))
model.add(Convolution2D(64,3,3,activation="elu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')  
#model = load_model('model.h5') #Preload the pre-trained weights
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*3, validation_data=validation_generator,   nb_val_samples=len(validation_samples), nb_epoch=3) #Make sure samples_per_epoch is 3 times the len(train_samples), because generator outputs 3*32 pictures instead of 32.
model.save('model.h5')