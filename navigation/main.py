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





class Drone:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z


def load_embeddings():
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
        
    class Decoder(nn.Module):
    
        def __init__(self, encoded_space_dim):
            super().__init__()
            self.decoder_lin = nn.Sequential(
                nn.Linear(encoded_space_dim, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 8 * 8 * 128),
                nn.ReLU(True)
            )

            self.unflatten = nn.Unflatten(dim=1, 
            unflattened_size=(128, 8, 8))

            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 
                stride=2, padding=0, output_padding=0),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 32, 4, 
                stride=2, padding=1, output_padding=0),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 16, 4, 
                stride=2, padding=0, output_padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 8, 4, stride=2, 
                padding=0, output_padding=0),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.ConvTranspose2d(8, 3, 4, stride=2, 
                padding=1, output_padding=0)
            )
            
        def forward(self, x):
            x = self.decoder_lin(x)
            x = self.unflatten(x)
            x = self.decoder_conv(x)
            x = torch.sigmoid(x)
            return x
        
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)
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
    decoder = Decoder(encoded_space_dim=256)
    best_decoder = torch.load('/media/kravtandr/SSD_1TB/Drone_nav/maks/best_decoder_02_08.pth')
    best_encoder = torch.load('/media/kravtandr/SSD_1TB/Drone_nav/maks/best_encoder_02_08.pth')
    encoder.load_state_dict(best_encoder)     
    decoder.load_state_dict(best_decoder)

    batch_size = 1
    train_dataset = CustomDataset(full_map=full_map[:, :5000], coord_dataset = coord_dataset, tile_size = tile_size, transform=transforms.ToTensor())
    val_dataset = CustomDataset(full_map=full_map[:, 5000:], coord_dataset = coord_dataset, tile_size = tile_size, transform=transforms.ToTensor())
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        i = 0
        for data in dataloader:
            if i == 1:
                break
            inputs, labels = data
            # print("Input: ", inputs)
            # print("Labels: ", labels)
            inputs = inputs.to(device)
            coded= encoder(inputs)
            outputs = decoder(encoder(inputs))
            inputs = inputs.cpu().numpy()
            coded = coded.cpu().numpy()
            # print(inputs, labels)
            #sleep(1)
            i+=1
            
            # print(coded)
            print(coded.shape)
            print("---------------")
            

    encoder.to(device)
    decoder.to(device)
    visualize_results(encoder, decoder, train_loader)

def main():
    # load emdeings (vectors)
    # embedings = []
    embedings = [[1,1,2],[2,0,2],[7,1,3],[7,4,100], [11,2222,1000]]
    
    # Преобразуем embeddings в DataFrame
    df = pd.DataFrame(embedings)

    # Вычисляем ковариационную матрицу для embedings и инвертируем её
    cov_matrix = np.cov(df.T)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    # drone start coords
    drone = Drone(0, 0, 0, 0)


    # while True:
    for i in range(1):
        # get image from camera
        fpv_image = None

        # image to encoder
        encoded_image = None

        min_mahalanobis_distance = np.inf
        # тут нужно потом сравнивать не со всеми а только с ближайшими
        for embeding_vector in embedings:
            mahalanobis_distance = distance.mahalanobis(encoded_image, embeding_vector, inv_cov_matrix)
            if mahalanobis_distance < min_mahalanobis_distance:
                min_mahalanobis_distance = mahalanobis_distance
                closest_embeding_vector = embeding_vector
        
        print(min_mahalanobis_distance, closest_embeding_vector)

        



if __name__ == '__main__':
    main()