import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net_V1(nn.Module):
    
    def __init__(self):
        super(Net_V1, self).__init__()
        

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
        self.drop_fc2 = nn.Dropout(p=0.7) #dropout must be extreme here
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        #initialize with xavier
#         I.xavier_uniform_(self.conv1.weight.data)
#         I.xavier_uniform_(self.conv2.weight.data)
#         I.xavier_uniform_(self.conv3.weight.data)
#         I.xavier_uniform_(self.conv4.weight.data)
#         I.xavier_uniform_(self.conv5.weight.data)
        I.xavier_uniform_(self.fc1.weight.data)
        I.xavier_uniform_(self.fc2.weight.data)
        I.xavier_uniform_(self.fc3.weight.data)
        
        #Inspired from a git code on yolact I read some time ago: https://github.com/dbolya/yolact/blob/master/yolact.py line 526. Also, they used it on conv weights but when I tried to train it locally the results were worse. So i let them on only fc layers

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




class Net_V2(nn.Module):

    def __init__(self):
        super(Net_V2, self).__init__()
        

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        #Suppose I have an input 224x224 and kernel 4x4
        self.conv1 = nn.Conv2d(1, 32, 3) #32 filters so image shape is (32,220x220)
        self.bn1   = nn.BatchNorm2d(num_features=32)

        self.conv2 = nn.Conv2d(32,64,3) #64 filters so image shape is (64,107,107)
        self.bn2   = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(64,128,3) #128 filters so image shape is (128, 51,51)
        self.bn3   = nn.BatchNorm2d(num_features=128)

        self.conv4 = nn.Conv2d(128,128,3) #256 filters so image shape is (256, 24,24)
        self.bn4   = nn.BatchNorm2d(num_features=128)  

        self.conv5 = nn.Conv2d(128,256,3) #256 filters so image shape is (512,12,12)
        self.bn5   = nn.BatchNorm2d(num_features=256)

        self.conv6 = nn.Conv2d(256,256,3) #256 filters so image shape is (512,12,12)
        self.bn6   = nn.BatchNorm2d(num_features=256)


        self.conv7 = nn.Conv2d(256,512,3) #512 filters so image shape is (512,12,12)
        self.bn7   = nn.BatchNorm2d(num_features=512)

        self.conv8 = nn.Conv2d(512,512,3) #512 filters so image shape is (512,12,12)
        self.bn8   = nn.BatchNorm2d(num_features=512)

        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3) 
        self.drop4 = nn.Dropout(p=0.4)
        self.drop_fc1_5 = nn.Dropout(p=0.5)
        self.drop_fc2_6 = nn.Dropout(p=0.6)

        self.pool = nn.MaxPool2d(2,2) #stride = 2 so image H&W is always halved

        self.fc1 = nn.Linear(51200 ,4096) #512*10*10 = 51200
        self.fc2 = nn.Linear(4096,1000)
        self.fc3 = nn.Linear(1000,136)
        # self.bn9 = nn.BatchNorm1d(num_features=4096) #1d for Dense layers
        # self.bn10 = nn.BatchNorm1d(num_features=1000) 
        # BatchNorm1d with batch size of 1 is problematic - has errors
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        #initialize with xavier

        I.xavier_uniform_(self.fc1.weight.data)
        I.xavier_uniform_(self.fc2.weight.data)
        I.xavier_uniform_(self.fc3.weight.data)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:


        # input 1 x 224x224
        x = self.bn1(F.relu(self.conv1(x))) # 32 x 222 x 222
        x = self.bn2(F.relu(self.conv2(x))) # 64 x 220 x 220
        x = self.pool(x) # 64 x 110 x 110

        x = self.bn3(F.relu(self.conv3(x))) # 128 x 108 x 108
        x = self.bn4(F.relu(self.conv4(x))) # 128 x 106 x 106
        x = self.pool(x) # 128 x 53 x 53

        x = self.bn5(F.relu(self.conv5(x))) # 256 x 51 x 51
        x = self.bn6(F.relu(self.conv6(x))) # 256 x 49 x 49
        x = self.pool(x) # 256 x 24 x 24

        x = self.bn7(F.relu(self.conv7(x))) # 512 x 22 x 22
        x = self.bn8(F.relu(self.conv8(x))) # 512 x 20 x 20
        x = self.pool(x) # 512 x 10 x 10 

        x = x.view(x.size(0), -1) #flatten
        # x = self.drop_fc1_5(self.bn9(F.relu(self.fc1(x))))
        # x = self.drop_fc2_6(self.bn10(F.relu(self.fc2(x))))
        # BatchNorm1d with batch size of 1 is problematic - has errors
        x = self.drop_fc1_5(F.relu(self.fc1(x)))
        x = self.drop_fc2_6(F.relu(self.fc2(x)))
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x



class Net_V3(nn.Module):
    
    def __init__(self):
        super(Net_V3, self).__init__()
        

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        #Suppose I have an input 224x224 and kernel 4x4
        self.conv1 = nn.Conv2d(1, 32, 3) 
        self.bn1   = nn.BatchNorm2d(num_features=32)

        self.conv2 = nn.Conv2d(32,64,3) 
        self.bn2   = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(64,128,3) 
        self.bn3   = nn.BatchNorm2d(num_features=128)

        self.conv5 = nn.Conv2d(128,256,3) 
        self.bn5   = nn.BatchNorm2d(num_features=256)


        self.conv7 = nn.Conv2d(256,512,3) 
        self.bn7   = nn.BatchNorm2d(num_features=512)

        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3) 
        self.drop4 = nn.Dropout(p=0.4)
        self.drop_fc1_5 = nn.Dropout(p=0.5)
        self.drop_fc2_6 = nn.Dropout(p=0.6)

        self.pool = nn.MaxPool2d(2,2) #stride = 2 so image H&W is always halved

        #self.fc1 = nn.Linear(51200 ,4096) #512*10*10 = 51200
        self.fc1 = nn.Linear(73728 ,4096)
        self.fc2 = nn.Linear(4096,1000)
        self.fc3 = nn.Linear(1000,136)
        # self.bn9 = nn.BatchNorm1d(num_features=4096) #1d for Dense layers
        # self.bn10 = nn.BatchNorm1d(num_features=1000) 
        # BatchNorm1d with batch size of 1 is problematic - has errors
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        #initialize with xavier

        I.xavier_uniform_(self.fc1.weight.data)
        I.xavier_uniform_(self.fc2.weight.data)
        I.xavier_uniform_(self.fc3.weight.data)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:


        # input 1 x 224x224
        x = self.bn1(F.relu(self.conv1(x))) # 32 x 222 x 222
        x = self.bn2(F.relu(self.conv2(x))) # 64 x 220 x 220
        x = self.pool(x) # 64 x 110 x 110

        x = self.bn3(F.relu(self.conv3(x))) # 128 x 108 x 108
        #x = self.bn4(F.relu(self.conv4(x))) 
        x = self.pool(x) # 128 x 54 x 54

        x = self.bn5(F.relu(self.conv5(x))) # 256 x 52 x 52
        #x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool(x) # 256 x 26 x 26

        x = self.bn7(F.relu(self.conv7(x))) # 512 x 24 x 24
        #x = self.bn8(F.relu(self.conv8(x))) # 512 x 20 x 20
        x = self.pool(x) # 512 x 12 x 12 

        x = x.view(x.size(0), -1) #flatten
        # x = self.drop_fc1_5(self.bn9(F.relu(self.fc1(x))))
        # x = self.drop_fc2_6(self.bn10(F.relu(self.fc2(x))))
        # BatchNorm1d with batch size of 1 is problematic - has errors
        x = self.drop_fc1_5(F.relu(self.fc1(x)))
        x = self.drop_fc2_6(F.relu(self.fc2(x)))
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x