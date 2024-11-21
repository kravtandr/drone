### Drone Navigation with Autoencoder

Проект по навигации дрона с использованием автоэнкодера для определения позиции на карте.


##### Структура проекта

```
DRONE/
├── map/                    # Директория с картой
│   └── test_map_crop.png  # Карта для навигации
├── models/                 # Обученные модели
│   ├── autoencoder_model.h5
│   └── best_encoder_02_08.pth
├── testing/               # Тестовые скрипты
│   ├── get_pos.py        # Получение позиции
│   └── test_camera.py    # Тестирование камеры
└── train/                 # Скрипты для обучения
```


##### Основные файлы

    1_prepare_dataset.ipynb - Генерация датасета из карты

    Создает тайлы из исходного изображения карты
    Внимание: может занять много места на диске при малом размере тайла
    2_train.ipynb - Обучение нейронной сети

    Обучение автоэнкодера на сгенерированном датасете
    3_prepare_emb_matx_test.ipynb - Подготовка эмбеддингов

    Сохранение эмбеддингов и матрицы для дальнейшего использования
    main.py - Основной файл для запуска системы навигации


##### Установка и запуск

###### Установите необходимые зависимости:

    pip install -r requirements.txt

###### Запустите последовательно jupyter notebooks в следующем порядке:

    1_prepare_dataset.ipynb
    2_train.ipynb
    3_prepare_emb_matx_test.ipynb

###### Запустите основной скрипт:

    python3 main.py

###### Важные файлы

    embeddings.pkl - Сохраненные эмбеддинги
    inv_cov_matrix.npy - Инверсная ковариационная матрица

##### Примечание

Папка navigation_workfiles содержит рабочие файлы и может быть проигнорирована


