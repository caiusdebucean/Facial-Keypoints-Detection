import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from models import Net
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor
import torch.optim as optim

PATH = './saved_models/keypoints_model_Adam_SL1_NaimishNet_partial_epoch_20.pt'
net = Net()
net.load_state_dict(torch.load(PATH))
net.eval()

weights1 = net.conv3.weight.data

w = weights1.numpy()

filter_index = 0

print(w[filter_index][0])
print(w[filter_index][0].shape)
plt.imshow(w[filter_index][0], cmap='gray')
plt.show()


data_transform = composed = transforms.Compose([Rescale(250),
                                                RandomCrop(224),
                                                Normalize(),
                                                ToTensor()])
transformed_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                             root_dir='./data/training/',
                                             transform=data_transform)

batch_size = 1

train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=1)

for batch_i, data in enumerate(train_loader):
    # get the input images and their corresponding labels
    images = data['image']
    key_pts = data['keypoints']
    image_np = images[0][0].numpy()
    image_convolved = cv2.filter2D(image_np, -1, w[filter_index][0])
    plt.imshow(image_np, cmap='gray')
    plt.show()
    plt.imshow(image_convolved, cmap='gray')
    plt.show()





# display the filter weights
plt.imshow(w[filter_index][0], cmap='gray')
plt.show()