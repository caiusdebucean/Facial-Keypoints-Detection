import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
from models import Net
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor
import torch.optim as optim


data_transform = composed = transforms.Compose([Rescale(250),
                                                RandomCrop(224),
                                                Normalize(),
                                                ToTensor()])
transformed_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                             root_dir='./data/training/',
                                             transform=data_transform)

batch_size = 32

train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=4)
test_dataset = FacialKeypointsDataset(csv_file='./data/test_frames_keypoints.csv',
                                             root_dir='./data/test/',
                                             transform=data_transform)

batch_size = 32

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
# net = nn.Sequential(
#     nn.Conv2d(1, 32, 4),
#     nn.MaxPool2d(2,2),
#     nn.Dropout(p=0.1),
#     nn.Conv2d(32,64,3),
#     nn.MaxPool2d(2,2),
#     nn.Dropout(p=0.2),
#     nn.Conv2d(64,128,2),
#     nn.MaxPool2d(2,2),
#     nn.Dropout(p=0.3),
#     nn.Conv2d(128,256,1),
#     nn.MaxPool2d(2,2),
#     nn.Dropout(p=0.4),
#     nn.Linear(43264 ,1000),
#     nn.Dropout(p=0.5),
#     nn.Linear(1000,1000),
#     nn.Dropout(p=0.5),
#     nn.Linear(1000,136))



# print(net)
# criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()
#tried MSELoss => loss value after 10 ep = not good enough
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

n_epochs = 15

##IMPORTANT##
# Because of the conversion to floats, we use .coda.FloatTensor


if net is not None:
    # prepare the net for training
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']
            
            images, key_pts = images.to(device), key_pts.to(device)

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            if device != 'cpu':
                key_pts = key_pts.type(torch.cuda.FloatTensor)
                images = images.type(torch.cuda.FloatTensor)
            else:
                key_pts = key_pts.type(torch.FloatTensor)
                images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0

    print('Finished Training')

#For testing we define the following functions

def net_sample_output():
    
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):
        
        # get sample data: images and ground truth keypoints
        images = sample['image'].to(device)
        key_pts = sample['keypoints'].to(device)

        # convert images to FloatTensors
        if device != 'cpu':
            print('GPU')
            images = images.type(torch.cuda.FloatTensor)
        else:
            images = images.type(torch.FloatTensor)
        # forward pass to get net output
        output_pts = net(images)
        
        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        
        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts

def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')

def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
    
    for i in range(batch_size):
        plt.figure(figsize=(20,10))
        ax = plt.subplot(1, batch_size, i+1)

        # un-transform the image data
        image = test_images[i].data   # get the image from it's Variable wrapper
        image = image.type(torch.FloatTensor)
        image = image.to('cpu')
        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.type(torch.FloatTensor)
        predicted_key_pts = predicted_key_pts.to('cpu')
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints  
        predicted_key_pts = predicted_key_pts*50.0+100
        
        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]         
            ground_truth_pts = ground_truth_pts*50.0+100
        
        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
            
        plt.axis('off')

    plt.show()


test_images, test_outputs, gt_pts = net_sample_output()
test_images, test_outputs, gt_pts = test_images.to('cpu'), test_outputs.to('cpu'), gt_pts.to('cpu')
print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())

visualize_output(test_images, test_outputs, gt_pts)


#Saving the model

## TODO: change the name to something uniqe for each new model
model_dir = 'saved_models/'
model_name = 'keypoints_model_Adam_SL1_NaimishNet_partial_xavier_conv'

# after training, save your model parameters in the dir 'saved_models'
torch.save(net.state_dict(), model_dir+model_name+'_epoch_'+str(n_epochs) + '.pt')


