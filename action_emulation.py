# action_emulation.py

import subprocess
import time
import logging
from object_priorities import object_priorities  # Импортируем object_priorities
from enum import Enum

# Настройка логирования
logging.basicConfig(level=logging.INFO)


# Определение действий с использованием Enum для повышения читаемости и предотвращения ошибок
class Action(Enum):
    SELECT_CARD = 0
    PRESS_OK = 1
    DO_NOTHING = 2
    CLICK_AT_COORDINATES = 3
    PRESS_CANCEL = 4
    DOUBLE_CLICK_AT_COORDINATES = 5


def perform_action(action, detected_objects, device_id, x=None, y=None):
    """
    Выполняет действие на устройстве на основе Action Enum и обнаруженных объектов.

    :param action: Член Enum Action, представляющий действие
    :param detected_objects: Список обнаруженных объектов
    :param device_id: Идентификатор устройства ADB
    :param x: Координата x для действия CLICK_AT_COORDINATES или DOUBLE_CLICK_AT_COORDINATES (по умолчанию None)
    :param y: Координата y для действия CLICK_AT_COORDINATES или DOUBLE_CLICK_AT_COORDINATES (по умолчанию None)
    """
    if not isinstance(action, Action):
        logging.error(
            f"Недопустимый тип для действия: {type(action)}. Ожидается член Enum Action."
        )
        return

    try:
        if action == Action.SELECT_CARD:
            # Выбираем карту с наивысшим приоритетом
            cards = [
                obj
                for obj in detected_objects
                if obj.get("class_name") in object_priorities
                and obj.get("class_name").lower() != "available"
            ]
            if cards:
                # Сортируем карты по приоритету, включая отрицательные
                cards.sort(
                    key=lambda obj: object_priorities.get(obj.get("class_name", ""), 0),
                    reverse=True,
                )
                selected_card = cards[0]
                x_coord = int(selected_card.get("x", 0))
                y_coord = int(selected_card.get("y", 0))
                priority = object_priorities.get(selected_card.get("class_name", ""), 0)
                if priority < 0:
                    logging.info(
                        f"Выбор карты {selected_card.get('class_name')} с отрицательным приоритетом на координатах ({x_coord}, {y_coord})"
                    )
                else:
                    logging.info(
                        f"Выбор карты с приоритетом: {selected_card.get('class_name')} на координатах ({x_coord}, {y_coord})"
                    )
                subprocess.run(
                    [
                        "adb",
                        "-s",
                        device_id,
                        "shell",
                        "input",
                        "tap",
                        str(x_coord),
                        str(y_coord),
                    ]
                )
                time.sleep(0.5)
            else:
                logging.warning("Карты для выбора не найдены.")

        elif action == Action.PRESS_OK:
            # Нажимаем кнопку 'OK' по фиксированным координатам
            x_coord = 1000  # Замените на реальные координаты кнопки 'OK'
            y_coord = 670  # Замените на реальные координаты кнопки 'OK'
            logging.info(
                f"Нажатие кнопки 'OK' на фиксированных координатах ({x_coord}, {y_coord})"
            )
            subprocess.run(
                [
                    "adb",
                    "-s",
                    device_id,
                    "shell",
                    "input",
                    "tap",
                    str(x_coord),
                    str(y_coord),
                ]
            )
            time.sleep(0.5)

        elif action == Action.CLICK_AT_COORDINATES:
            if x is not None and y is not None:
                # Кликаем по заданным координатам
                logging.info(
                    f"Выполнение клика по координатам: x={x}, y={y} на устройстве {device_id}"
                )
                subprocess.run(
                    [
                        "adb",
                        "-s",
                        device_id,
                        "shell",
                        "input",
                        "tap",
                        str(int(x)),
                        str(int(y)),
                    ]
                )
                time.sleep(
                    1
                )  # Увеличенная задержка после клика для обеспечения выполнения
            else:
                logging.warning(
                    "Координаты не заданы для действия CLICK_AT_COORDINATES."
                )

        elif action == Action.DOUBLE_CLICK_AT_COORDINATES:
            if x is not None and y is not None:
                # Выполняем двойной клик по заданным координатам
                logging.info(
                    f"Выполнение двойного клика по координатам: x={x}, y={y} на устройстве {device_id}"
                )
                subprocess.run(
                    [
                        "adb",
                        "-s",
                        device_id,
                        "shell",
                        "input",
                        "tap",
                        str(int(x)),
                        str(int(y)),
                    ]
                )
                time.sleep(0.1)  # Короткая задержка между кликами
                subprocess.run(
                    [
                        "adb",
                        "-s",
                        device_id,
                        "shell",
                        "input",
                        "tap",
                        str(int(x)),
                        str(int(y)),
                    ]
                )
                time.sleep(0.5)
            else:
                logging.warning(
                    "Координаты не заданы для действия DOUBLE_CLICK_AT_COORDINATES."
                )

        elif action == Action.PRESS_CANCEL:
            # Нажимаем кнопку 'Cancel' по фиксированным координатам
            x_coord = 1000  # Замените на реальные координаты кнопки 'Cancel'
            y_coord = 850  # Замените на реальные координаты кнопки 'Cancel'
            logging.info(
                f"Нажатие кнопки 'Cancel' на фиксированных координатах ({x_coord}, {y_coord})"
            )
            subprocess.run(
                [
                    "adb",
                    "-s",
                    device_id,
                    "shell",
                    "input",
                    "tap",
                    str(x_coord),
                    str(y_coord),
                ]
            )
            time.sleep(0.5)

        elif action == Action.DO_NOTHING:
            # Ничего не делаем
            logging.info("Действие 'Ничего не делать'")
            time.sleep(0.5)

        else:
            logging.warning(f"Неизвестное действие: {action}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Ошибка при выполнении команды adb: {e}")
    except Exception as e:
        logging.error(f"Произошла непредвиденная ошибка: {e}")
