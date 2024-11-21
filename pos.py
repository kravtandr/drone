from scipy.spatial import distance
import numpy as np 
import pandas as pd 
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


train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Drone:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z

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

def load_embeddings():
    image_path = './map/test_map_crop.png'
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

    best_encoder = torch.load('./models/best_encoder_02_08.pth', weights_only=True)
    encoder.load_state_dict(best_encoder)   
      

    batch_size = 1
    train_dataset = CustomDataset(full_map=full_map[:, :5000], coord_dataset = coord_dataset, tile_size = tile_size, transform=transforms.ToTensor())
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    encoder.to(device)
    encoder.eval()
    embedings = []
    labels = []
    with torch.no_grad():
        for data in dataloader:
            # if i == 1:
            #     break
            inputs, label = data
            # print("Input: ", inputs)
            # print("Labels: ", labels)
            # print("Input: ", inputs.shape)
            inputs = inputs.to(device)
            coded= encoder(inputs)
    
            inputs = inputs.cpu().numpy().flatten().tolist()
            coded = coded.cpu().numpy().flatten().tolist()

            
            # print(coded)
            # print(coded.shape)
            embedings.append(coded)
            labels.append(label)
            # print("---------------")
            

    print("Emb Amount: ", len(embedings))
    print("Emb: ", embedings[0:3], "...")
    return labels, embedings



def image_to_tensor(drone):
    # пока использую статичное изображение, потом буду получать от камеры
    # step = 50
    tile_size = 300
    x = drone.x
    y = drone.y
    full_map_image_path = './map/test_map_crop.png'
    full_map = cv2.imread(full_map_image_path)
    fpv_image = full_map[y:y+tile_size, x:x+tile_size]
    fpv_image = cv2.cvtColor(fpv_image, cv2.COLOR_BGR2RGB)
    transform=transforms.ToTensor()
    fpv_image = transform(fpv_image)
    fpv_image = fpv_image.unsqueeze(0)
    print("Image shape:",fpv_image.shape)
    return fpv_image

def image_to_embeding(inputs):
     
    encoder = Encoder(encoded_space_dim=256)
    encoder.to(device)
    best_encoder = torch.load('./models/best_encoder_02_08.pth')
    encoder.load_state_dict(best_encoder)  
    encoder.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        coded= encoder(inputs)
        coded = coded.cpu().numpy()
        print("Image coded emb shape:",coded.shape)
        return coded


# загржуам эмбединги в память
labels, data = load_embeddings()


# ключ-значение для меток
embedings = dict(zip(map(tuple,data), labels))

# Преобразуем embeddings в DataFrame
df = pd.DataFrame(data)
print(df.shape)


# Вычисляем ковариационную матрицу для embedings и инвертируем её
cov_matrix = np.cov(df.T)
inv_cov_matrix = np.linalg.inv(cov_matrix)

# drone start coords
drone = Drone(0, 333, 777, 0)

def nearbyAreaCheck(drone, embedings, embeding_vector):
    nearbyDelta = 300
    x = embedings[tuple(embeding_vector)][0]
    y = embedings[tuple(embeding_vector)][1]
    if abs(x-drone.x) < nearbyDelta and abs(y-drone.y) < nearbyDelta:
        return True
    else:
        return False
    

# while True:
for i in range(1):
    # get image from camera
    fpv_image = image_to_tensor(drone)

    # image to encoder
    encoded_image = image_to_embeding(fpv_image).flatten()

    min_mahalanobis_distance = np.inf
    closest_embeding_vector = None
    # тут нужно потом сравнивать не со всеми а только с ближайшими
    for embeding_vector in data:
        if nearbyAreaCheck(drone, embedings, embeding_vector):
            mahalanobis_distance = distance.mahalanobis(encoded_image, embeding_vector, inv_cov_matrix)
            # print(mahalanobis_distance)
            if mahalanobis_distance < min_mahalanobis_distance:
                min_mahalanobis_distance = mahalanobis_distance
                closest_embeding_vector = embeding_vector

    print("Min mahal: ",min_mahalanobis_distance)
    print("New Drone coords: ",embedings[tuple(closest_embeding_vector)])
    print("Real Drone coords: ", drone.x, drone.y)