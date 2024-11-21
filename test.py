from picamera2 import Picamera2
from datetime import datetime
import time
import os

class RaspberryCamera:
    """
    Класс для работы с камерой Raspberry Pi.
    Предоставляет базовый функционал для получения и сохранения изображений.
    """
    
    def __init__(self, resolution=(1920, 1080)):
        """
        Инициализация камеры с заданным разрешением.
        
        Args:
            resolution (tuple): Кортеж (ширина, высота) для установки разрешения камеры
        """
        self.camera = None
        self.resolution = resolution
        self.is_initialized = False
    
    def initialize(self):
        """
        Инициализация и настройка камеры.
        
        Returns:
            bool: True если инициализация прошла успешно, False в случае ошибки
        """
        try:
            self.camera = Picamera2()
            
            # Настройка конфигурации камеры
            config = self.camera.create_still_configuration(
                main={"size": self.resolution}
            )
            self.camera.configure(config)
            
            # Запуск камеры
            self.camera.start()
            
            # Даем камере время на инициализацию
            time.sleep(2)
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Ошибка при инициализации камеры: {str(e)}")
            return False
    
    def capture_image(self):
        """
        Получение одиночного кадра с камеры.
        
        Returns:
            numpy.ndarray: Изображение в формате numpy array или None в случае ошибки
        """
        if not self.is_initialized:
            print("Камера не инициализирована")
            return None
            
        try:
            return self.camera.capture_array()
        except Exception as e:
            print(f"Ошибка при получении изображения: {str(e)}")
            return None
    
    def save_image(self, image, directory="images"):
        """
        Сохранение полученного изображения в файл.
        
        Args:
            image: Изображение для сохранения
            directory (str): Директория для сохранения изображений
            
        Returns:
            str: Путь к сохраненному файлу или None в случае ошибки
        """
        try:
            # Создаем директорию, если она не существует
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # Генерируем имя файла на основе текущего времени
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{directory}/image_{timestamp}.jpg"
            
            # Сохраняем изображение
            self.camera.capture_file(filename)
            return filename
            
        except Exception as e:
            print(f"Ошибка при сохранении изображения: {str(e)}")
            return None
    
    def close(self):
        """
        Корректное закрытие камеры и освобождение ресурсов.
        """
        if self.is_initialized and self.camera:
            try:
                self.camera.close()
                self.is_initialized = False
            except Exception as e:
                print(f"Ошибка при закрытии камеры: {str(e)}")

    def __enter__(self):
        """
        Метод для поддержки контекстного менеджера (with statement).
        """
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Метод для поддержки контекстного менеджера (with statement).
        """
        self.close()


from scipy.spatial import distance
import numpy as np 
import pandas as pd 
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import pickle

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available')
else:
    print('CUDA is available!')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
TILE_STEP = 50
TILE_SIZE = 300
ENCODED_SPACE_DIM = 256
NEARBY_DELTA = 300
MAP_PATH = './map/test_map_crop.png'
ENCODER_MODEL_PATH = './models/best_encoder_02_08.pth'
EMBEDDINGS_PATH = "embeddings.pkl"
MATRIX_PATH = "inv_cov_matrix.npy"

def load_matrix(filename):
    """
    Загружает матрицу из файла
    
    Параметры:
    filename: str - имя файла для загрузки (с расширением .npy)
    
    Возвращает:
    numpy array - загруженная матрица
    """
    try:
        matrix = np.load(filename)
        print(f"Матрица успешно загружена из файла {filename}")
        return matrix
    except Exception as e:
        print(f"Ошибка при загрузке матрицы: {e}")
        return None
    
def load_embeddings_from_file(filename):
    try:
        with open(filename, 'rb') as file:
            embeddings = pickle.load(file)
        print(f"Эмбеддинги успешно загружены из файла {filename}")
        return embeddings
    except Exception as e:
        print(f"Ошибка при загрузке эмбеддингов: {e}")
        return None

class Drone:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z

class Encoder(nn.Module):

    def __init__(self, ENCODED_SPACE_DIM):
        super().__init__()
        
        self.encoded_space_dim = ENCODED_SPACE_DIM
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
            nn.Linear(1024, ENCODED_SPACE_DIM)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x 

def image_to_tensor(image):
    fpv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform=transforms.ToTensor()
    fpv_image = transform(fpv_image)
    fpv_image = fpv_image.unsqueeze(0)
    # print("Image shape:",fpv_image.shape)
    return fpv_image

def image_to_embeding(inputs):
     
    encoder = Encoder(ENCODED_SPACE_DIM=256)
    encoder.to(device)
    best_encoder = torch.load(ENCODER_MODEL_PATH, weights_only=True, map_location=device("cpu"))
    encoder.load_state_dict(best_encoder)  
    encoder.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        coded= encoder(inputs)
        coded = coded.cpu().numpy()
        print("Image coded emb shape:",coded.shape)
        return coded

def nearbyAreaCheck(drone, embedings, embeding_vector):
    x = embedings[tuple(embeding_vector)][0]
    y = embedings[tuple(embeding_vector)][1]
    if abs(x-drone.x) < NEARBY_DELTA and abs(y-drone.y) < NEARBY_DELTA:
        return True
    else:
        return False



# drone start coords
drone = Drone(0, 333, 777, 0)

embedings = load_embeddings_from_file(EMBEDDINGS_PATH) 
inv_cov_matrix = load_matrix(MATRIX_PATH)

# while True:
for i in range(1):
    # get image from camera
    with RaspberryCamera() as camera:
        # Получаем изображение
        image = camera.capture_image()
        if image is not None:
            # Сохраняем изображение
            saved_path = camera.save_image(image)
            if saved_path:
                print(f"Изображение успешно сохранено: {saved_path}")
            else:
                print("Ошибка при сохранении изображения")
        else:
            print("Ошибка при получении изображения")
    
    fpv_image = image_to_tensor(image)

    # image to encoder
    encoded_image = image_to_embeding(fpv_image).flatten()

    min_mahalanobis_distance = np.inf
    closest_embeding_vector = None
    # тут нужно потом сравнивать не со всеми а только с ближайшими
    for embeding_vector in  embedings.keys():
        if nearbyAreaCheck(drone, embedings, embeding_vector):
            mahalanobis_distance = distance.mahalanobis(encoded_image, embeding_vector, inv_cov_matrix)
            # print(mahalanobis_distance)
            if mahalanobis_distance < min_mahalanobis_distance:
                min_mahalanobis_distance = mahalanobis_distance
                closest_embeding_vector = embeding_vector

    print("Min mahal: ",min_mahalanobis_distance)
    print("New Drone coords: ",embedings[tuple(closest_embeding_vector)])
    print("Real Drone coords: ", drone.x, drone.y)