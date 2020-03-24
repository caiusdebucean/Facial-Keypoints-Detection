import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        #Suppose I have an input 224x224 and kernel 4x4
        self.conv1 = nn.Conv2d(1, 32, 5) #32 filters so image shape is (32,220x220)
        self.pool1 = nn.MaxPool2d(2,2) #stride = 2 so image shape is (32,110,110) - I think it will get grounded to 110 from 110.5 - TODO: test it
        self.drop1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv2d(32,64,4) #64 filters so image shape is (64,107,107)
        self.pool2 = nn.MaxPool2d(2,2) #stride = 2 so image shape is (64,53,53)
        self.drop2 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv2d(64,128,3) #128 filters so image shape is (128, 51,51)
        self.pool3 = nn.MaxPool2d(2,2) #stride = 2 so image shape is (128,25,25)
        self.drop3 = nn.Dropout(p=0.3)        
        self.conv4 = nn.Conv2d(128,256,2) #256 filters so image shape is (256, 24,24)
        self.pool4 = nn.MaxPool2d(2,2) #stride = 2 so image shape is (256,12,12)
        self.drop4 = nn.Dropout(p=0.4)
        self.conv5 = nn.Conv2d(256,512,1) #256 filters so image shape is (512,12,12)
        self.pool5 = nn.MaxPool2d(2,2) #stride = 2 so image shape is (512,6,6)
        self.drop5 = nn.Dropout(p=0.5)
        #TODO OPTIONAL: Add batchnorm

        # self.fc1 = nn.Linear(43264 ,1000) #256*13*13 = 43264 
        self.fc1 = nn.Linear(18432 ,1000) #512*6*6 = 4321843264 
        self.fc2 = nn.Linear(1000,1000)
        self.fc3 = nn.Linear(1000,136)

        self.drop_fc1 = nn.Dropout(p=0.6)
        self.drop_fc2 = nn.Dropout(p=0.7)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        #initialize with xavier
        I.xavier_uniform_(self.conv1.weight.data)
        I.xavier_uniform_(self.conv2.weight.data)
        I.xavier_uniform_(self.conv3.weight.data)
        I.xavier_uniform_(self.conv4.weight.data)
        I.xavier_uniform_(self.conv5.weight.data)
        I.xavier_uniform_(self.fc1.weight.data)
        I.xavier_uniform_(self.fc2.weight.data)
        I.xavier_uniform_(self.fc3.weight.data)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))
        x = self.drop4(self.pool4(F.relu(self.conv4(x))))
        x = self.drop5(self.pool5(F.relu(self.conv5(x))))


        x = x.view(x.size(0), -1)
        x = self.drop_fc1(F.relu(self.fc1(x)))
        x = self.drop_fc2(F.relu(self.fc2(x)))
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
