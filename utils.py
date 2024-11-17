# utils.py

import cv2
import os
import logging
import math
from object_priorities import object_priorities
import tensorflow as tf
from tensorflow.keras.layers import Layer
import json
import shutil
import datetime


def preprocess_image(image_path):
    """
    Предварительная обработка изображения: загрузка, преобразование в градации серого, изменение размера и нормализация.
    """
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Не удалось загрузить изображение {image_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (84, 84)) / 255.0
    return img


class CombineStreams(Layer):
    """
    Пользовательский слой для объединения потоков Value и Advantage в Dueling DQN.
    """

    def __init__(self, **kwargs):
        super(CombineStreams, self).__init__(**kwargs)

    def call(self, inputs):
        value, advantage = inputs
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

    def get_config(self):
        config = super(CombineStreams, self).get_config()
        return config


def save_training_params(filepath, params):
    """
    Сохраняет параметры обучения в JSON файл.
    """
    try:
        with open(filepath, "w") as f:
            json.dump(params, f, indent=4)
        logging.info(f"Параметры обучения сохранены в {filepath}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении параметров обучения: {e}")


def load_training_params(filepath):
    """
    Загружает параметры обучения из JSON файла.
    """
    try:
        with open(filepath, "r") as f:
            params = json.load(f)
        logging.info(f"Параметры обучения загружены из {filepath}")
        return params
    except Exception as e:
        logging.error(f"Ошибка при загрузке параметров обучения: {e}")
        return {}


def objects_equal(obj_list1, obj_list2, tolerance=10):
    """
    Сравнивает два списка обнаруженных объектов.
    """
    if len(obj_list1) != len(obj_list2):
        return False

    # Сортируем списки по class_name, x и y для последовательного сравнения
    sorted1 = sorted(obj_list1, key=lambda x: (x["class_name"], x["x"], x["y"]))
    sorted2 = sorted(obj_list2, key=lambda x: (x["class_name"], x["x"], x["y"]))

    for obj1, obj2 in zip(sorted1, sorted2):
        if obj1["class_name"] != obj2["class_name"]:
            return False
        if (
            abs(obj1["x"] - obj2["x"]) > tolerance
            or abs(obj1["y"] - obj2["y"]) > tolerance
        ):
            return False

    return True


def calculate_distance(x, y, center_x, center_y):
    """
    Вычисляет евклидово расстояние между точкой (x, y) и центром (center_x, center_y).
    """
    return math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)


def get_sorted_available_coordinates(
    detected_objects, center_x, center_y, image_width, image_height, max_coords=3
):
    """
    Сортирует доступные объекты по расстоянию до центра и приоритету.
    Возвращает список координат для кликов.
    """
    available_objects = [
        obj for obj in detected_objects if obj["class_name"].lower() == "available"
    ]

    # Вычисляем расстояние до центра для каждого объекта
    for obj in available_objects:
        obj["distance"] = calculate_distance(
            obj["x"] * image_width, obj["y"] * image_height, center_x, center_y
        )

    # Сортируем по расстоянию и приоритету
    sorted_objects = sorted(
        available_objects,
        key=lambda obj: (obj["distance"], -object_priorities.get(obj["class_name"], 0)),
    )

    # Выбираем топ-3 или все, если их меньше трех
    selected_objects = (
        sorted_objects[:max_coords]
        if len(sorted_objects) > max_coords
        else sorted_objects
    )

    # Извлекаем координаты
    sorted_coordinates = [
        (int(obj["x"] * image_width), int(obj["y"] * image_height))
        for obj in selected_objects
    ]

    return sorted_coordinates


def save_training_screenshot(episode, step, training_screenshots_dir):
    """
    Сохраняет текущий скриншот для обучения YOLO.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    screenshot_filename = f"episode_{episode}_step_{step}_{timestamp}.png"
    screenshot_path = os.path.join(training_screenshots_dir, screenshot_filename)
    try:
        shutil.copy("screen.png", screenshot_path)
        logging.info(f"Скриншот сохранен для обучения YOLO: {screenshot_path}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении скриншота для обучения YOLO: {e}")
        if not os.path.exists("screen.png"):
            logging.error("Файл screen.png не найден. Скриншот не сохранен.")
            return

