# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 represents the wished output video
* Udacity-Full-Run.mp4 represents my drive in birds-eye-perspective
* Udacity-Full-Run99.avi represents the birds-eye-perspective with 99fps
* in GPU mode, there is a folder called run1 with all my recorded jpgs of my run

I tried to drive that car, but I am a really bad driver or the machine reacts to slow!
Therefore, I used your recordings for my analysis.


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed



My model is an immitation of the shown net of the tutorial video of nVidia. It worked perfect for me!
But I extended the net a little bit.
[image1]: ./images/nVidia_net.png "nVidia NET"


The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 
I first use a Keras Lambda function with the shape of 160x320 and RGB depth (3)
Then I crop the images, as described in the tutorial (70,25) to remove the cars hood and the content above the street.
Then I haven start with my convolution neural network with three 5x5 kernels. I have the same output space as mentioned in nVidias model.
Then I followup with two layers of 3x3 kernels. 
After that, I include a MaxPooling2D function to reduce overfitting a little bit.
After that, I include a Dropout function to reduce the output size again and prevent overfitting.
Then I flatten my output for my fully connected network.
To classify my output I follow with four fully connnected layers.



#### 2. Attempts to reduce overfitting in the model

As mentioned before, in line 117 and 118, I included MaxPooling2D and Dropout (drob possibility of 50%) function to prevent overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 126).
For my batch size I chose 32. Again, the keep propability of my dropout function is 50% or 0.5.
I chose to run 3 epochs because of a timing problem. Each computation took me 10minutes, even on your machines.


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 
Thats true! As mentioned above, I was not a good driver. I tried to drive that car, but even I could not hold it on the road. Therefore, I used your already recorded images.
I first started to only use the center pictures. Unfortunately, the car was not able to take tighter road curves and stay on the road.
So, I used more training data as mentioned in the tutorial lessons and included the side pictures right and left, as well. Additionally, I added the mentioned correction of 0.2 to heach measurement angle.
For details about how I created the training data, see the next section. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the nVidia model. But this information is given already above!

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. 


The final step was to run the simulator to see how well the car was driving around track one. 
There were a few spots where the vehicle fell off the track.
Especially, in tighter road curves, the car was not able to stay on the road. Again, I had to use additional training data and images.
Here I chose to use the left and the right camera for it.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
The resulting video is uploaded. See file run1.mp4 for driver perspective and Udacity-Full-Run.mp4 for birds-eye-perspective.

#### 2. Final Model Architecture
I will not describe it again, because it is already said in the sectoin above.


#### 3. Creation of the Training Set & Training Process

First of my processing, I load my data samples.
Second, I used train_test_split function and put 20% of the data into a validation set. 
 
Then my two generator parts are beginning for training and validation. Each generator is processing in the same way.
At the start of my generator function, I randomly shuffled the data set, as shown in the lessons.

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 
Here is an example image of center lane driving:
[image2]: ./images/2021_02_06_22_50_22_953.jpg "Drivers Perspective"

For training (again) I used your dataset. The different perspectives helped me to gather more data for my training (3 times).
For me the most important thing is to handle tighter road curves that my network learns how to steer back to the middle of the road.
[image3]: ./images/center_2016_12_01_13_30_48_287.jpg "Center Recording"
[image4]: ./images/left_2016_12_01_13_30_48_287.jpg "Left Recording"
[image5]: ./images/right_2016_12_01_13_30_48_287.jpg "Right Recording"

To augment the data sat, I also flipped images and angles.
I used that feature in my generator function. Because our track is counterclockwise, there is always more training data for left curves.
To have a quite fair training, I flipped each picture and angle and added them to my training set.
Additionally, I gather more training data.

The ideal number of epochs was 3. I tried once 5 epochs which would reslut in better results (but small).
I used an adam optimizer so that manually training the learning rate wasn't necessary.


Looking to my training process, I used 6428 samples in each of my 3 epochs. 
My training loss was continously decresing to a level of 0.0031 which is quite good I suppose.
The validatoin loss i decresing as well to a level of 0.0222 which is good, as well.
No losses increased over time.

Epoch 1/3
6428/6428 [==============================] - 730s 114ms/step - loss: 0.0134 - val_loss: 0.0245
Epoch 2/3
6428/6428 [==============================] - 718s 112ms/step - loss: 0.0049 - val_loss: 0.0241
Epoch 3/3
6428/6428 [==============================] - 725s 113ms/step - loss: 0.0031 - val_loss: 0.0222


### 4. Challenges beside of the project

The first problem was, that the training took 30minutes for each complete run (in the Udacity environment. 
As a workaround I tried to run that training on my local (I think powerful) machine. Unfortunately, tensorflow-gpu 1.4.0 keras-gpu 2.06 does not want to run on my local GPU. I use Visual Studio Code. 
Maybe you can help to tell me the reason it will not accept my gpu as device. I tried nearly everything. I have an Nvidia Quadro RTX 3000. And I installed all that Cuda stuff, just to be save...

----
Furhtermore, I got an error to create to create the video with FFmpeg at the final stage...
+udacity RuntimeError: No ffmpeg exe could be found. Install ffmpeg on your system, or set the IMAGEIO_FFMPEG_EXE environment variable.

I needed to fix this an install ffmpeg
sudo apt install software-properties-common
sudo apt update
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt install ffmpeg
ffmpeg -version


### 5. Final Output Log
(/opt/carnd_p3/behavioral) root@40bfa2ea38c9:/home/workspace/CarND-Behavioral-Cloning-P3# python model.py
Using TensorFlow backend.

model.py:111: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(24, (5, 5), activation="relu", strides=(2, 2))`
  model.add(Conv2D(24,5,5,subsample=(2,2),activation='relu'))
model.py:112: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(36, (5, 5), activation="relu", strides=(2, 2))`
  model.add(Conv2D(36,5,5,subsample=(2,2),activation='relu'))
model.py:113: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(48, (5, 5), activation="relu", strides=(2, 2))`
  model.add(Conv2D(48,5,5,subsample=(2,2),activation='relu'))
model.py:115: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(64, (3, 3), activation="relu")`
  model.add(Conv2D(64,3,3,activation='relu'))
model.py:116: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(64, (3, 3), activation="relu")`
  model.add(Conv2D(64,3,3,activation='relu'))
model.py:117: UserWarning: Update your `MaxPooling2D` call to the Keras 2 API: `MaxPooling2D(data_format="channels_first", pool_size=(2, 2))`
  model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
2021-02-06 22:05:44.981091: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2021-02-06 22:05:44.981143: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2021-02-06 22:05:44.981156: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2021-02-06 22:05:44.981169: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2021-02-06 22:05:44.981184: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2021-02-06 22:05:45.062266: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-02-06 22:05:45.063152: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:00:04.0
Total memory: 11.17GiB
Free memory: 11.10GiB
2021-02-06 22:05:45.063228: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2021-02-06 22:05:45.063262: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2021-02-06 22:05:45.063294: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)
model.py:142: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(generator=<generator..., validation_steps=51.0, steps_per_epoch=6428, validation_data=<generator..., verbose=1, epochs=3)`
  model.fit_generator(generator=train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, validation_steps=np.ceil(len(validation_samples)/batch_size), epochs=num_epochs, verbose=1)  # make the number of epochs adjustable, verbose shows the output
Epoch 1/3
6428/6428 [==============================] - 730s 114ms/step - loss: 0.0134 - val_loss: 0.0245
Epoch 2/3
6428/6428 [==============================] - 718s 112ms/step - loss: 0.0049 - val_loss: 0.0241
Epoch 3/3
6428/6428 [==============================] - 725s 113ms/step - loss: 0.0031 - val_loss: 0.0222