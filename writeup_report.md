#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used the PilotNet model from nVidia that has been popular among the CarND students (model.py lines 67-80). I used PilotNet, because it has been engineered especially for autonomous steering.
This is a computationally heavy model with 5 convolutional layers and 4 dense layers, so in order to process 320x160 images from the dataset, this model requires a GPU for an acceptable training time.
Although I didn't use a GPU, I managed very short training times (around 100 sec.) per epoch by resizing the image in the lambda layer of the model.
In order to introduce nonlinearity to the convolutional layers, I was going to use RELU's which are very popular, however as I read through the udacity forums and other online sources, I found out that people are having better steering results with ELUs. There are papers analyzing ELUs by comparing to RELUs, so ELU was my choice of activation function in this model. (https://arxiv.org/abs/1511.07289)


####2. Attempts to reduce overfitting in the model

My model did not overfit with both training and validation loss being around 0.03. Unlike the previous projects, I did not need to do any kind of image augmentation or flipping as suggested by udacity forum users and mentors. 

However I'd like to introduce a few of the concepts I've come across online. If the model overfits, I'd first introduce dropout layers starting from the bottom dense layers one by one, and increase the number of epochs accordingly, since  dropping out dense layer neurons will need more training runs than usual.

One more approach is to collect more data at places where the car struggles to make a pass, like the bridge and the dirt road entrance. You can record data to a few passes, and add it to your dataset, and train your model again. It is advised to save your model.h5 file, which saves the weights too, and load it before the next network training. Doing this will update the weights, instead of training them from scratch, and will save you significant time.

Image augmentation like flipping images would be useful too, which is kind of driving the track backwards. Augmentations like brightness adjustment can be done as well. I only did normalization of the images for this project.

The model was trained and validated on udacity data set and did not overfit. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Please watch run1.mp4 included in this folder.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 80).

####4. Appropriate training data

I tried to collect training data myself, however not being a regular gamer did not help and I couldn't keep the car on the road with the keyboard or the mouse. So I opted for using udacity's track 1 data.

For training, I used center, left and right camera images at the same time with steering correction added/subtracted to left and right camera images.

###Model Architecture and Training Strategy

####1. Solution Design Approach

Choosing a model architecture is hard, since there is no hard coded rules of thumb. The deeper the network goes, more computationally intense it gets at the same time. I chose PilotNet,  because it was designed to drive a car autonomously. Its 5 layer deep convolutional layers can learn many features on the road with enough training data, while last 5 dense layers adjust the steering according to the learned features. The model is complicated so the input images need to be resized before being fed to the model.

In order to gauge how well the model was working, I split (1/5 of the samples are set as validation set) my image and steering angle data into a training and validation set. Loss of the training and validation sets are almost equal at around 0.03 with 3 epochs. So this is a sign that the model is not overfitting or underfitting. Using left and right camera angles tripled the training data, therefore helped the model overcome overfitting problems before arising.

The final step was to run the simulator to see how well the car was driving around track one. It drove beautifully without getting out of track, including the bridge and the dirt road entrance at the first try.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...
**Cropping2D Layer:** Crop top 70 pixels(mostly skyline), and bottom 25 pixels(hood of the car)
**Lambda Layer**: Call resize and normalize function, which outputs 64x64 images
**Conv2D Layer:** 5x5 kernel, 24 filters, 2x2 strides
**ELU Activation function**
**Conv2D Layer:** 5x5 kernel, 36 filters, 2x2 strides
**ELU Activation function**
**Conv2D Layer:** 5x5 kernel, 48 filters, 2x2 strides
**ELU Activation function**
**Conv2D Layer:** 3x3 kernel, 64 filters,
**ELU Activation function**
**Conv2D Layer:** 3x3 kernel, 64 filters,
**ELU Activation function**
**Flatten layer**
**Dense (FC):** Output size: 100
**Dense (FC):** Output size: 50
**Dense (FC):** Output size: 10
**Dense (FC):** Output size: 1

####3. Creation of the Training Set & Training Process

First I tried to take some training data myself, but apparently I'm not really good with games, so I failed to keep the car in the center line most of the time. So I decided to use the data provided by udacity carnd program.

If my model couldn't finish the track successfully, then I would try recording some recovery data while driving from sides to the center of the lane. Also, I would definitely add driving data at the different locations.

Moreover, the data augmentation improves the model when used in moderation. Flipping images, adding slight angles and shifting the image are popular augmentation techniques in order to train better motels.

After the collection process, I had around 8000 number of data points. My model did the normalization inside the lambda layer by the function called "normalize and resize". This dataset is not huge however, I decided to train my model with all of the the center, left and right pictures, making the dataset 24000. 

General idea about shuffling the dataset generator was generally towards removing most of 0 degree steering data, in order to eliminate the straight driving bias. I did not agree with this, because even when the car is going straight, left and right angles will have a positive and negative bias respectively, therefore training the model more robustly. by repating the same scene from two different points of view.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.  I used 3 epochs at the start, which was around 0.03. I used an adam optimizer so that manually training the learning rate wasn't necessary.

You can watch the autonomous ride over [here](/video.mp4).
