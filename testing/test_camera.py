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


from time import sleep

def basic_example():
    """
    Базовый пример использования камеры: получение и сохранение одного снимка
    """
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

def continuous_capture_example(interval=5, duration=30):
    """
    Пример непрерывной съемки с заданным интервалом
    
    Args:
        interval (int): Интервал между снимками в секундах
        duration (int): Общая продолжительность съемки в секундах
    """
    with RaspberryCamera(resolution=(1280, 720)) as camera:
        start_time = time.time()
        captured_count = 0
        
        print(f"Начинаем съемку на {duration} секунд с интервалом {interval} секунд")
        
        while (time.time() - start_time) < duration:
            image = camera.capture_image()
            if image is not None:
                saved_path = camera.save_image(image)
                if saved_path:
                    captured_count += 1
                    print(f"Сохранено изображение {captured_count}: {saved_path}")
                else:
                    print("Ошибка при сохранении изображения")
            else:
                print("Ошибка при получении изображения")
            
            # Ждем до следующего снимка
            sleep(interval)
        
        print(f"Съемка завершена. Всего сохранено изображений: {captured_count}")

def main():
    """
    Главная функция с демонстрацией различных примеров использования камеры
    """
    print("1. Демонстрация получения одиночного снимка:")
    basic_example()
    
    print("\n2. Демонстрация непрерывной съемки:")
    continuous_capture_example(interval=2, duration=10)

if __name__ == "__main__":
    main()