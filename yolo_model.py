# yolo_model.py

import os
from ultralytics import YOLO
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Путь к вашей локальной модели YOLOv8
local_model_path = r"D:\Game2\DATASETimg\YOLOv8\runs\detect\train9\weights\best.pt"

if not os.path.exists(local_model_path):
    raise FileNotFoundError(
        f"Локальная модель YOLOv8 не найдена по пути: {local_model_path}"
    )

try:
    # Загрузка локальной модели YOLOv8
    yolo_model = YOLO(local_model_path)

    logging.info("Локальная модель YOLOv8 успешно загружена и готова к использованию")
except Exception as e:
    logging.error(f"Ошибка при загрузке локальной модели YOLOv8: {e}")
    yolo_model = None
