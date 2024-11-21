import os
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms as T
from torchvision import transforms
import pytorch_msssim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from torch.optim import lr_scheduler
import gc
from time import sleep
import math

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        self.encoded_space_dim = encoded_space_dim
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 4, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 4, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(8 * 8 * 128, 1024),
            nn.ReLU(True),
            nn.Linear(1024, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        #z_mean = self.encoder_lin(x)
        #z_log_var = self.encoder_lin(x)
        #N = torch.normal(0, 1, size=(z_log_var.size()[0], self.encoded_space_dim))
        #N = N.to(device)
        #print('N: ', N.shape, ' z_log_var: ', z_log_var.shape)
        return x #(torch.exp(z_log_var / 2) * N + z_mean), z_mean, z_log_var
    
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


image_path = '/media/kravtandr/SSD_1TB/Drone_nav/map/test/test_map_crop.png'
full_map = cv2.imread(image_path)
full_map = cv2.cvtColor(full_map, cv2.COLOR_BGR2RGB)
# координаты тайлов
height, width, _ = full_map.shape
step = 50
tile_size = 300
coord_dataset = pd.DataFrame(columns = ['x', 'y'])
for x in range(0, width - tile_size, step):
    for y in range(0, height - tile_size, step):
        coord_dataset.loc[len(coord_dataset.index)] = [x, y]
class CustomDataset(Dataset):
    def __init__(self, full_map, coord_dataset, tile_size, transform=None):
        self.mapa = full_map
        self.coords = coord_dataset
        self.tile_size = tile_size
        self.transform = transform

    def __len__(self):
        return len(self.coords.index)

    def __getitem__(self, idx):
        x = self.coords.iloc[idx]['x']
        y = self.coords.iloc[idx]['y']
        image = full_map[y:y+tile_size, x:x+tile_size]
        if self.transform:
            image = self.transform(image)
            
        return image, (x, y)

encoder = Encoder(encoded_space_dim=256)

best_decoder = torch.load('/media/kravtandr/SSD_1TB/Drone_nav/maks/best_decoder_02_08.pth')
best_encoder = torch.load('/media/kravtandr/SSD_1TB/Drone_nav/maks/best_encoder_02_08.pth')
encoder.load_state_dict(best_encoder)     

batch_size = 128
train_dataset = CustomDataset(full_map=full_map[:, :5000], coord_dataset = coord_dataset, tile_size = tile_size, transform=transforms.ToTensor())
val_dataset = CustomDataset(full_map=full_map[:, 5000:], coord_dataset = coord_dataset, tile_size = tile_size, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def visualize_results(encoder, dataloader):
    encoder.eval()
    with torch.no_grad():
        i = 0
        for data in dataloader:
            if i == 1:
                break
            inputs, labels = data
            print("Input: ", inputs)
            print("Labels: ", labels)
            inputs = inputs.to(device)
            coded= encoder(inputs)
            inputs = inputs.cpu().numpy()
            coded = coded.cpu().numpy()
            # print(inputs, labels)
            #sleep(1)
            i+=1
            
            print(coded)
            print("---------------")
            

encoder.to(device)
visualize_results(encoder, train_loader)