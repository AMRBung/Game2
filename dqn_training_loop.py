# dqn_training_loop.py

import random
import os
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import logging
import time
import numpy as np
from collections import deque
import cv2
from adb_screenshot_capture import capture_screenshot, get_device_id
from dqn_agent_model import build_dueling_dqn
from action_emulation import perform_action, Action
from object_priorities import object_priorities
from yolo_model import yolo_model
from utils import (
    preprocess_image,
    CombineStreams,
    save_training_params,
    load_training_params,
    objects_equal,
    save_training_screenshot,
    get_sorted_available_coordinates,
)
from game_utils import calculate_reward, get_available_cards
import threading  # Импорт для работы с потоками

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Создание глобальной блокировки для доступа к screen.png
screenshot_lock = threading.Lock()

# Параметры сохранения скриншотов обучения
SAVE_TRAINING_SCREENSHOTS = (
    False  # Установите False True, чтобы отключить сохранение скриншотов
)

# Максимальное количество неудачных попыток перед понижением приоритета
MAX_FAILED_ATTEMPTS = 3

# Интервал времени (в секундах) для восстановления приоритета
PRIORITY_RESET_INTERVAL = 300  # Например, 5 минут

# Исходные приоритеты карт (для восстановления)
initial_object_priorities = object_priorities.copy()

# Директория для логов TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Директория для сохранения моделей
saved_models_dir = "saved_models"
os.makedirs(saved_models_dir, exist_ok=True)

# Директория для сохранения скриншотов для обучения YOLO
training_screenshots_dir = "training_screenshots"
os.makedirs(training_screenshots_dir, exist_ok=True)
logging.info(
    f"Директория для скриншотов обучения YOLO: {training_screenshots_dir}")

# Параметры обучения
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
batch_size = 64  # Увеличенный размер батча
memory_size = 20000  # Увеличенный объем памяти
memory = deque(maxlen=memory_size)

# Количество действий
n_actions = len(Action)  # Используем количество членов Enum

# Порог уверенности для предсказаний YOLO
CONFIDENCE_THRESHOLD = 0.4

# Идентификатор устройства ADB
device_id = get_device_id()
logging.info(f"Используется устройство ADB с идентификатором: {device_id}")

# Размер клетки для генерации случайных кликов вокруг базы
CELL_WIDTH = 250
CELL_HEIGHT = 250

# Глобальный словарь для отслеживания неудачных попыток
failed_cards = {}

# Окружающие клетки
surrounding_cells = [(1, 1), (1, 2), (1, 3), (2, 1),
                     (2, 3), (3, 1), (3, 2), (3, 3)]
logging.info(f"Окружающие клетки: {surrounding_cells}")

# Специальные классы, которые требуют немедленного клика без нажатия 'OK'
SPECIAL_CLASSES = [
    "battle",
    "battleback",
    "battledown",
    "battleok",
    "battleokay",
    "battletry",
    "continuegame",
    "umulti",
    "uinferno",
]


def get_random_click_position(cell_row, cell_col):
    """
    Генерирует случайные координаты внутри заданной клетки.
    """
    x_start = cell_col * CELL_WIDTH
    y_start = cell_row * CELL_HEIGHT

    click_x = random.randint(x_start, x_start + CELL_WIDTH - 1)
    click_y = random.randint(y_start, y_start + CELL_HEIGHT - 1)

    return click_x, click_y


def lower_priority(class_name):
    """
    Понижает приоритет указанного класса на 1, но не ниже минимального значения.
    """
    current_priority = object_priorities.get(class_name, 0)
    min_priority = -10  # Минимальный приоритет
    new_priority = max(current_priority - 1, min_priority)
    object_priorities[class_name] = new_priority
    logging.info(f"Понижение приоритета карты {class_name} до {new_priority}")


def periodic_screenshot(device_id, interval=2):
    """
    Периодически снимает скриншоты устройства с заданным интервалом.
    """
    while True:
        try:
            with screenshot_lock:
                capture_screenshot(device_id)
        except Exception as error:
            logging.error(f"Ошибка при снятии скриншота: {error}")
        time.sleep(interval)


def capture_and_preprocess(device_id, state):
    """
    Захватывает скриншот и предобрабатывает изображение для получения следующего состояния.
    """
    with screenshot_lock:
        capture_screenshot(device_id)
        frame = preprocess_image("screen.png")
    if frame is None:
        logging.error("Не удалось загрузить новое состояние, пропуск шага")
        return None
    next_state = np.append(
        state[:, :, 1:], np.expand_dims(frame, axis=2), axis=2)
    return next_state


# Инициализация модели Dueling DQN
input_shape = (84, 84, 4)

# Проверяем, существует ли сохраненная модель и параметры
saved_model_path = os.path.join(
    saved_models_dir, "dqn_model_interrupted.keras")
saved_params_path = os.path.join(
    saved_models_dir, "dqn_model_interrupted_params.json")

if os.path.exists(saved_model_path) and os.path.exists(saved_params_path):
    try:
        dqn_model = tf.keras.models.load_model(
            saved_model_path,
            custom_objects={"CombineStreams": CombineStreams, "tf": tf},
        )
        logging.info(
            f"Загружена сохраненная модель из файла {saved_model_path}")

        # Загружаем параметры обучения
        training_params = load_training_params(saved_params_path)
        epsilon = training_params.get("epsilon", epsilon)
        gamma = training_params.get("gamma", gamma)
        epsilon_min = training_params.get("epsilon_min", epsilon_min)
        epsilon_decay = training_params.get("epsilon_decay", epsilon_decay)
        logging.info(f"Параметры обучения обновлены из {saved_params_path}")
    except Exception as error:
        logging.error(f"Ошибка при загрузке модели или параметров: {error}")
        logging.info("Создана новая модель Dueling DQN")
        dqn_model = build_dueling_dqn(input_shape, n_actions)
else:
    dqn_model = build_dueling_dqn(input_shape, n_actions)
    logging.info("Создана новая модель Dueling DQN")

# Инициализируем целевую модель
target_dqn_model = build_dueling_dqn(input_shape, n_actions)
target_dqn_model.set_weights(dqn_model.get_weights())


def update_target_model():
    """
    Обновляет веса целевой модели, копируя веса из основной модели.
    """
    target_dqn_model.set_weights(dqn_model.get_weights())


def train_dqn_model():
    """
    Обучает модель DQN на основе случайной выборки из памяти.
    """
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states = np.array([experience[0] for experience in batch])
    actions = np.array(
        [experience[1].value for experience in batch]
    )  # Используем .value для получения int
    rewards = np.array([experience[2] for experience in batch])
    next_states = np.array([experience[3] for experience in batch])
    dones = np.array([experience[4] for experience in batch])

    # Double DQN
    q_values_next = dqn_model.predict(next_states)
    next_actions = np.argmax(q_values_next, axis=1)
    target_q_values_next = target_dqn_model.predict(next_states)
    target_q_values = (
        rewards
        + (1 - dones)
        * gamma
        * target_q_values_next[np.arange(batch_size), next_actions]
    )

    q_values = dqn_model.predict(states)
    q_values[np.arange(batch_size), actions] = target_q_values

    # Обучение модели
    dqn_model.fit(
        states, q_values, epochs=1, verbose=0, callbacks=[tensorboard_callback]
    )


# Машина состояний
class AgentState:
    WAITING_FOR_CARD = 0
    SELECTING_CARD = 1
    PRESSING_OK = 2


def save_model_and_params(model, params, save_dir):
    """
    Сохраняет модель и параметры обучения в указанные директории.
    """
    # Сохранение модели под фиксированным именем для прерывания
    interrupted_model_path = os.path.join(
        save_dir, "dqn_model_interrupted.keras")
    model.save(interrupted_model_path, save_format="keras")
    logging.info(
        f"Модель сохранена как {interrupted_model_path} для прерывания")

    # Сохранение параметров обучения
    interrupted_params_path = os.path.join(
        save_dir, "dqn_model_interrupted_params.json"
    )
    save_training_params(interrupted_params_path, params)
    logging.info(
        f"Параметры обучения сохранены как {interrupted_params_path} для прерывания"
    )


# Запуск потока для периодического снятия скриншотов
screenshot_thread = threading.Thread(
    target=periodic_screenshot, args=(device_id,))
# Поток завершится при завершении основной программы
screenshot_thread.daemon = True
screenshot_thread.start()

# Инициализация переменной 'episode'
episode = 0

# Основной цикл обучения агента
try:
    for episode in range(1, 1001):
        logging.info(f"Начало эпизода {episode}")

        # Получаем начальное состояние из четырех кадров
        with screenshot_lock:
            capture_screenshot(device_id)
            frame = preprocess_image("screen.png")
            img = cv2.imread("screen.png")  # Добавлено в блок с lock
        if frame is None or img is None:
            logging.error(
                "Не удалось получить начальное состояние, пропуск эпизода")
            continue
        state = np.stack([frame] * 4, axis=2)
        done = False
        total_reward = 0
        step = 0
        max_steps = 50  # Увеличено количество шагов в одном эпизоде
        prev_detected_objects = []
        steps_without_progress = 0
        prev_wave = 0  # Инициализация текущей волны
        agent_state = AgentState.WAITING_FOR_CARD
        selected_card = None  # Инициализация selected_card

        # Переменная для отслеживания попыток нажатия "OK"
        press_ok_attempts = 0
        max_press_ok_attempts = 2

        # Переменная для отслеживания отсутствия изменений на экране
        no_change_counter = 0
        max_no_change = 5  # Количество итераций без изменений перед кликом

        # Счётчик динамических кликов
        dynamic_click_counter = 0
        max_dynamic_clicks = (
            3  # После трех динамических кликов выполнять фиксированный клик
        )

        # Получаем размеры изображения
        if img is not None:
            image_height, image_width = img.shape[:2]
            logging.info(
                f"Скриншот имеет размер: {image_width}x{image_height}")
        else:
            logging.error(
                "Не удалось загрузить скриншот для определения размера.")
            continue  # Пропустить текущий эпизод, если скриншот не загружен

        center_x = image_width / 2
        center_y = image_height / 2

        while not done and step < max_steps:
            step += 1

            # Выполняем инференс с использованием YOLO для анализа состояния
            with screenshot_lock:
                capture_screenshot(device_id)
                results = yolo_model(
                    "screen.png", conf=CONFIDENCE_THRESHOLD, verbose=False
                )

            # Обрабатываем каждый объект предсказания
            detected_objects = []
            if results:
                predictions = results[
                    0
                ].boxes  # Получаем боксы первого (и единственного) изображения
                if predictions:
                    for pred in predictions:
                        class_id = int(
                            pred.cls.cpu().numpy()[0]
                        )  # Извлекаем первый элемент
                        class_name = yolo_model.names[class_id]
                        # Координаты bounding box
                        bbox = pred.xyxy.cpu().numpy()[0]
                        confidence = float(
                            pred.conf.cpu().numpy()[0]
                        )  # Извлекаем первый элемент
                        x_center = (bbox[0] + bbox[2]) / 2
                        y_center = (bbox[1] + bbox[3]) / 2
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        detected_objects.append(
                            {
                                "class_name": class_name,
                                "x": x_center / image_width,  # Нормализуем координаты
                                "y": y_center / image_height,  # Нормализуем координаты
                                "confidence": confidence,
                                "priority": object_priorities.get(class_name, 0),
                            }
                        )
                else:
                    logging.info("Объекты не обнаружены на изображении.")
                    detected_objects = []
            else:
                logging.error(
                    "Ошибка при выполнении инференса с моделью YOLOv8.")
                detected_objects = []

            # Сравниваем текущие обнаруженные объекты с предыдущими
            if objects_equal(detected_objects, prev_detected_objects):
                no_change_counter += 1
                logging.info(f"Итерация без изменений: {no_change_counter}")
                if no_change_counter >= max_no_change:
                    # Динамически обновляем FIXED_CLICK_COORDINATES
                    FIXED_CLICK_COORDINATES = get_sorted_available_coordinates(
                        detected_objects,
                        center_x,
                        center_y,
                        image_width,
                        image_height,
                        max_coords=3,
                    )
                    logging.info(
                        f"Динамически обновленные координаты для кликов: {FIXED_CLICK_COORDINATES}"
                    )

                    if FIXED_CLICK_COORDINATES:
                        # Если есть доступные динамические координаты
                        click_x, click_y = random.choice(
                            FIXED_CLICK_COORDINATES)
                        logging.info(
                            f"Скрин не менялся в течение {max_no_change} итераций. Кликаем по динамическим координатам ({click_x}, {click_y})"
                        )
                    else:
                        # Если нет доступных динамических координат, используйте фиксированные
                        click_x, click_y = random.choice(
                            [(260, 1800), (540, 1800), (830, 1800), (1000, 800)]
                        )
                        logging.info(
                            f"Скрин не менялся в течение {max_no_change} итераций. Кликаем по фиксированным координатам ({click_x}, {click_y})"
                        )

                    perform_action(
                        Action.CLICK_AT_COORDINATES,
                        detected_objects,
                        device_id,
                        click_x,
                        click_y,
                    )
                    action = Action.CLICK_AT_COORDINATES  # Присваиваем действие

                    # Получаем новое состояние после действия
                    next_state = capture_and_preprocess(device_id, state)
                    if next_state is None:
                        logging.error(
                            "Не удалось загрузить новое состояние, пропуск шага"
                        )
                        continue

                    # Добавляем негативное вознаграждение за отсутствие изменений
                    reward = -1  # Негативное вознаграждение
                    total_reward += reward

                    # Сохраняем переход в память
                    memory.append((state, action, reward, next_state, done))

                    # Обучение модели
                    train_dqn_model()

                    # Обновление состояния
                    state = next_state
                    prev_detected_objects = detected_objects
                    no_change_counter = 0  # Сброс счётчика

                    # Увеличиваем счётчик динамических кликов
                    dynamic_click_counter += 1
                    logging.info(
                        f"Счётчик динамических кликов: {dynamic_click_counter}"
                    )

                    # Проверяем, достиг ли счётчик трёх кликов
                    if dynamic_click_counter >= max_dynamic_clicks:
                        # Выполняем клик по фиксированным координатам
                        click_x_fixed, click_y_fixed = random.choice(
                            [(260, 1800), (540, 1800), (830, 1800), (1000, 800)]
                        )
                        logging.info(
                            f"Три динамических клика выполнены. Кликаем по фиксированным координатам ({click_x_fixed}, {click_y_fixed})"
                        )
                        perform_action(
                            Action.CLICK_AT_COORDINATES,
                            detected_objects,
                            device_id,
                            click_x_fixed,
                            click_y_fixed,
                        )
                        action = Action.CLICK_AT_COORDINATES  # Присваиваем действие

                        # Получаем новое состояние после действия
                        next_state = capture_and_preprocess(device_id, state)
                        if next_state is None:
                            logging.error(
                                "Не удалось загрузить новое состояние, пропуск шага"
                            )
                            continue

                        # Добавляем негативное вознаграждение за использование фиксированного клика
                        reward = (
                            -1
                        )  # Можно изменить на другое значение, если необходимо
                        total_reward += reward

                        # Сохраняем переход в память
                        memory.append(
                            (state, action, reward, next_state, done))

                        # Обучение модели
                        train_dqn_model()

                        # Обновление состояния
                        state = next_state
                        prev_detected_objects = detected_objects

                        # Сброс счётчика динамических кликов
                        dynamic_click_counter = 0
                        continue  # Переходим к следующей итерации
            else:
                no_change_counter = 0  # Сброс счётчика при обнаружении изменений

            # Обновляем предыдущие обнаруженные объекты
            prev_detected_objects = detected_objects.copy()

            # Определяем доступные карты, отсортированные по приоритету
            available_cards = get_available_cards(
                detected_objects,
                failed_cards,
                object_priorities,
                initial_object_priorities,
                MAX_FAILED_ATTEMPTS,
                PRIORITY_RESET_INTERVAL,
            )
            logging.info(
                f"Доступные карты (отсортированы по приоритету): {[card['class_name'] for card in available_cards]}"
            )

            # Машина состояний
            if agent_state == AgentState.WAITING_FOR_CARD:
                if available_cards:
                    agent_state = AgentState.SELECTING_CARD
                else:
                    logging.info("Ожидание появления карт для выбора.")
                    time.sleep(1)
                    continue  # Переходим к следующей итерации

            if agent_state == AgentState.SELECTING_CARD:
                if not available_cards:
                    logging.info("Нет доступных карт для выбора.")
                    agent_state = AgentState.WAITING_FOR_CARD
                    continue

                # Выбор карты с наивысшим приоритетом
                selected_card = available_cards[0]
                card_class_lower = selected_card["class_name"].lower()

                # Проверка, не было ли слишком много неудачных попыток для этой карты
                if (
                    card_class_lower in failed_cards
                    and failed_cards[card_class_lower]["attempts"]
                    >= MAX_FAILED_ATTEMPTS
                ):
                    logging.info(
                        f"Пропуск выбора карты {selected_card['class_name']} из-за слишком большого количества неудачных попыток."
                    )
                    # Удаляем карту из списка доступных
                    available_cards = available_cards[1:]
                    if not available_cards:
                        logging.info(
                            "Нет доступных карт после пропуска неудачных. Переход в состояние ожидания."
                        )
                        agent_state = AgentState.WAITING_FOR_CARD
                        continue
                    selected_card = available_cards[0]
                    card_class_lower = selected_card["class_name"].lower()

                selected_card_priority = object_priorities.get(
                    selected_card["class_name"], 0
                )
                logging.info(
                    f"Выбор карты {selected_card['class_name']} с приоритетом {selected_card_priority}"
                )

                # Определяем координаты карты (в пикселях)
                card_class = selected_card["class_name"]
                card_position = {
                    "x": selected_card["x"], "y": selected_card["y"]}

                # Переводим нормализованные координаты в пиксели
                click_x = int(card_position["x"] * image_width)
                click_y = int(card_position["y"] * image_height)
                logging.info(
                    f"Кликаем по карте {card_class} на координатах ({click_x}, {click_y})"
                )

                # **Сохранение скриншота перед выбором карты**
                if SAVE_TRAINING_SCREENSHOTS:
                    save_training_screenshot(
                        episode, step, training_screenshots_dir)

                perform_action(
                    Action.CLICK_AT_COORDINATES,
                    detected_objects,
                    device_id,
                    click_x,
                    click_y,
                )
                action = Action.CLICK_AT_COORDINATES  # Присваиваем действие

                # Проверяем, является ли выбранная карта одной из специальных
                if card_class_lower in SPECIAL_CLASSES:
                    # Для специальных классов выполняем только клик без нажатия 'OK'
                    action = Action.CLICK_AT_COORDINATES  # Присваиваем действие
                    # Не переходим в состояние PRESSING_OK

                    # Получаем новое состояние после действия
                    next_state = capture_and_preprocess(device_id, state)
                    if next_state is None:
                        logging.error(
                            "Не удалось загрузить новое состояние, пропуск шага"
                        )
                        continue

                    # Расчет вознаграждения
                    reward, steps_without_progress, prev_wave = calculate_reward(
                        detected_objects=detected_objects,
                        steps_without_progress=steps_without_progress,
                        prev_wave=prev_wave,
                        selected_card=selected_card,
                    )
                    total_reward += reward

                    # Сохраняем переход в память
                    memory.append((state, action, reward, next_state, done))

                    # Обучение модели
                    train_dqn_model()

                    # Обновление состояния
                    state = next_state
                    prev_detected_objects = detected_objects

                elif card_class_lower == "fastwave":
                    # Выполняем двойной клик по "fastwave"
                    perform_action(
                        Action.DOUBLE_CLICK_AT_COORDINATES,
                        detected_objects,
                        device_id,
                        click_x,
                        click_y,
                    )
                    action = Action.DOUBLE_CLICK_AT_COORDINATES  # Присваиваем действие
                    # Возможно, после двойного клика не требуется переход в состояние PRESSING_OK

                    # Получаем новое состояние после действия
                    next_state = capture_and_preprocess(device_id, state)
                    if next_state is None:
                        logging.error(
                            "Не удалось загрузить новое состояние, пропуск шага"
                        )
                        continue

                    # Расчет вознаграждения
                    reward, steps_without_progress, prev_wave = calculate_reward(
                        detected_objects=detected_objects,
                        steps_without_progress=steps_without_progress,
                        prev_wave=prev_wave,
                        selected_card=selected_card,
                    )
                    total_reward += reward

                    # Сохраняем переход в память
                    memory.append((state, action, reward, next_state, done))

                    # Обучение модели
                    train_dqn_model()

                    # Обновление состояния
                    state = next_state
                    prev_detected_objects = detected_objects

                else:
                    # Нажимаем кнопку 'OK' после клика по карте
                    perform_action(Action.PRESS_OK,
                                   detected_objects, device_id)
                    action = Action.PRESS_OK  # Присваиваем действие
                    agent_state = (
                        AgentState.PRESSING_OK
                    )  # Переход в состояние PRESSING_OK

                    # Получаем новое состояние после действия
                    next_state = capture_and_preprocess(device_id, state)
                    if next_state is None:
                        logging.error(
                            "Не удалось загрузить новое состояние, пропуск шага"
                        )
                        continue

                    # Расчет вознаграждения
                    reward, steps_without_progress, prev_wave = calculate_reward(
                        detected_objects=detected_objects,
                        steps_without_progress=steps_without_progress,
                        prev_wave=prev_wave,
                        selected_card=selected_card,
                    )
                    total_reward += reward

                    # Сохраняем переход в память
                    memory.append((state, action, reward, next_state, done))

                    # Обучение модели
                    train_dqn_model()

                    # Обновление состояния
                    state = next_state
                    prev_detected_objects = detected_objects

            elif agent_state == AgentState.PRESSING_OK:
                # Проверяем, доступна ли кнопка 'OK'
                ok_detected = any(
                    obj["class_name"].lower() == "ok" for obj in detected_objects
                )
                if ok_detected:
                    # Нажимаем кнопку 'OK'
                    perform_action(Action.PRESS_OK,
                                   detected_objects, device_id)
                    action = Action.PRESS_OK  # Присваиваем действие
                    press_ok_attempts = 0  # Сброс попыток
                    agent_state = AgentState.WAITING_FOR_CARD

                    # Сброс информации о неудачных попытках для карты
                    if selected_card:
                        card_class_lower = selected_card["class_name"].lower()
                        if card_class_lower in failed_cards:
                            del failed_cards[card_class_lower]
                            object_priorities[selected_card["class_name"]] = (
                                initial_object_priorities[selected_card["class_name"]]
                            )
                            logging.info(
                                f"Приоритет карты {selected_card['class_name']} восстановлен после успешной попытки."
                            )

                    # **Сохранение скриншота только при успешном нажатии 'OK'**
                    if SAVE_TRAINING_SCREENSHOTS:
                        save_training_screenshot(
                            episode, step, training_screenshots_dir
                        )

                    # Получаем новое состояние после действия
                    next_state = capture_and_preprocess(device_id, state)
                    if next_state is None:
                        logging.error(
                            "Не удалось загрузить новое состояние, пропуск шага"
                        )
                        continue

                    # Расчет вознаграждения
                    reward, steps_without_progress, prev_wave = calculate_reward(
                        detected_objects=detected_objects,
                        steps_without_progress=steps_without_progress,
                        prev_wave=prev_wave,
                        selected_card=selected_card,
                    )
                    total_reward += reward

                    # Сохраняем переход в память
                    memory.append((state, action, reward, next_state, done))

                    # Обучение модели
                    train_dqn_model()

                    # Обновление состояния
                    state = next_state
                    prev_detected_objects = detected_objects

                else:
                    if press_ok_attempts < max_press_ok_attempts:
                        # Кликаем случайно внутри окружающих клеток рядом с базой
                        cell_row, cell_col = random.choice(surrounding_cells)
                        click_x, click_y = get_random_click_position(
                            cell_row, cell_col)
                        logging.info(
                            f"Попытка нажать 'OK' не удалась. Попытка {press_ok_attempts + 1} из {max_press_ok_attempts}. Кликаем по координатам ({click_x}, {click_y})"
                        )
                        perform_action(
                            Action.CLICK_AT_COORDINATES,
                            detected_objects,
                            device_id,
                            click_x,
                            click_y,
                        )
                        action = Action.CLICK_AT_COORDINATES  # Присваиваем действие
                        press_ok_attempts += 1
                        time.sleep(0.5)  # Задержка между попытками

                        # Получаем новое состояние после действия
                        next_state = capture_and_preprocess(device_id, state)
                        if next_state is None:
                            logging.error(
                                "Не удалось загрузить новое состояние, пропуск шага"
                            )
                            continue

                        # Расчет вознаграждения
                        reward, steps_without_progress, prev_wave = calculate_reward(
                            detected_objects=detected_objects,
                            steps_without_progress=steps_without_progress,
                            prev_wave=prev_wave,
                            selected_card=selected_card,
                        )
                        total_reward += reward

                        # Сохраняем переход в память
                        memory.append(
                            (state, action, reward, next_state, done))

                        # Обучение модели
                        train_dqn_model()

                        # Обновление состояния
                        state = next_state
                        prev_detected_objects = detected_objects
                    else:
                        logging.info(
                            "Не удалось нажать 'OK' после 2 попыток. Нажимаем 'Cancel' и выбираем другую карту."
                        )
                        perform_action(Action.PRESS_CANCEL,
                                       detected_objects, device_id)
                        action = Action.PRESS_CANCEL  # Присваиваем действие
                        press_ok_attempts = 0  # Сброс попыток
                        agent_state = AgentState.SELECTING_CARD

                        # Понижение приоритета выбранной карты
                        if selected_card:
                            card_class_lower = selected_card["class_name"].lower(
                            )
                            if card_class_lower not in failed_cards:
                                failed_cards[card_class_lower] = {
                                    "attempts": 1,
                                    "last_failed_time": time.time(),
                                }
                            else:
                                failed_cards[card_class_lower]["attempts"] += 1
                                failed_cards[card_class_lower][
                                    "last_failed_time"
                                ] = time.time()

                            # Понижение приоритета после определённого количества неудач
                            if (
                                failed_cards[card_class_lower]["attempts"]
                                >= MAX_FAILED_ATTEMPTS
                            ):
                                lower_priority(selected_card["class_name"])

                        # Получаем новое состояние после действия
                        next_state = capture_and_preprocess(device_id, state)
                        if next_state is None:
                            logging.error(
                                "Не удалось загрузить новое состояние, пропуск шага"
                            )
                            continue

                        # Расчет вознаграждения
                        reward, steps_without_progress, prev_wave = calculate_reward(
                            detected_objects=detected_objects,
                            steps_without_progress=steps_without_progress,
                            prev_wave=prev_wave,
                            selected_card=selected_card,
                        )
                        total_reward += reward

                        # Сохраняем переход в память
                        memory.append(
                            (state, action, reward, next_state, done))

                        # Обучение модели
                        train_dqn_model()

                        # Обновление состояния
                        state = next_state
                        prev_detected_objects = detected_objects

                # Обновление epsilon и целевой модели
                if agent_state != AgentState.PRESSING_OK:
                    if epsilon > epsilon_min:
                        epsilon *= epsilon_decay

                # Обновление целевой модели каждые 100 шагов
                if step % 100 == 0:
                    update_target_model()

        logging.info(
            f"Эпизод {episode} завершен. Набрано вознаграждение: {total_reward}"
        )

except KeyboardInterrupt:
    # Сохранение модели при прерывании обучения пользователем
    current_params = {
        "epsilon": epsilon,
        "gamma": gamma,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        # Добавьте другие параметры по необходимости
    }
    save_model_and_params(dqn_model, current_params, saved_models_dir)
    logging.info("Завершение программы.")
