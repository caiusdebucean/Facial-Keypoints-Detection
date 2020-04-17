# Facial-Keypoints-Detection

A Neural network trained YouTube Faces Dataset on  to detect the facial keypoints from images. This is the first project of the [*Udacity Computer Vision Expert Nanodegree*](https://www.udacity.com/course/computer-vision-nanodegree--nd891).

### Overview

This method uses **Haar Cascades** in images in order to locate _RoIs_ (Region of interests), which are then given to a **CNN** to generate facial keypoints. 
The CNN architecture is pictured below:

![CNN architecture](https://i.imgur.com/jwxtQ8C.png)

For a more comprehensive understanding of this project, unzip the jupyter notebooks found in the `Notebooks` directory and iterate through them.

### Requirements

The dataset used can be found [here](https://www.cs.tau.ac.il/~wolf/ytfaces/)

You can download the dataset from this [link](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip)
Put the contents in the _data_ directory

The following _libraries_ are mandatory: _torch, torchvision, matplotlib, numpy, pandas, cv2, pillow_ 


### Run the network

To train and test the whole pipeline, run the following command:

>python network.py 

### Results

The image below is an example of how this network isolates a face region and then applies the one pair of the predicted facial keypoints. 

![Obama and Michelle](https://i.imgur.com/YWq5F8o.png)

This model is very simple and should be treated as a quick solution to detecting points of interest on human faces.

Debucean Caius-Ioan @Udacity
