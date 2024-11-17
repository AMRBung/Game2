# game_utils.py

import time
import logging
from typing import List, Dict, Any, Tuple
from object_priorities import object_priorities


def get_current_wave(detected_objects: List[Dict[str, Any]]) -> int:
    waves = [obj for obj in detected_objects if obj["class_name"].startswith("w")]
    if not waves:
        return 0
    wave_numbers = []
    for obj in waves:
        # Используем лямбда-функцию для предиката
        wave_str = "".join(filter(lambda char: char.isdigit(), obj["class_name"]))
        if wave_str:
            wave_numbers.append(int(wave_str))
    current_wave = max(wave_numbers) if wave_numbers else 0
    logging.info(f"Текущая волна: {current_wave}")
    return current_wave


def calculate_reward(
    detected_objects: List[Dict[str, Any]],
    steps_without_progress: int,
    prev_wave: int,
    selected_card: Dict[str, Any] = None,
) -> Tuple[float, int, int]:
    reward = 0.0

    # Получаем текущую волну
    current_wave = get_current_wave(detected_objects)

    # Вознаграждение за переход на новую волну
    if current_wave > prev_wave:
        reward += 50  # Фиксированное вознаграждение за новую волну
        logging.info(f"Переход на волну {current_wave}, вознаграждение: {reward}")
        prev_wave = current_wave
        steps_without_progress = 0  # Сброс счетчика
    else:
        steps_without_progress += 1

    # Штраф за отсутствие прогресса (не новый уровень)
    if steps_without_progress >= 10:
        reward -= 5
        logging.info(
            f"Отсутствие прогресса в течение {steps_without_progress} шагов, штраф: -5"
        )
        steps_without_progress = 0  # Сброс счетчика

    # Штраф за выбор карты с отрицательным приоритетом, исключая 'available'
    if selected_card:
        class_name = selected_card["class_name"]
        class_name_lower = class_name.strip().lower()
        priority = object_priorities.get(class_name, 0)
        if priority < 0 and class_name_lower != "available":
            reward -= 2
            logging.info(
                f"Выбор карты {class_name} с отрицательным приоритетом, штраф: -2"
            )

    return reward, steps_without_progress, prev_wave


def get_available_cards(
    detected_objects: List[Dict[str, Any]],
    failed_cards: Dict[str, Dict[str, Any]],
    obj_priorities: Dict[str, int],
    initial_obj_priorities: Dict[str, int],
    max_failed_attempts: int,
    priority_reset_interval: int,
) -> List[Dict[str, Any]]:
    """
    Извлекает доступные карты из обнаруженных объектов и сортирует их по приоритету.
    Исключает только временно пониженные карты.
    """
    current_time = time.time()
    cards = []
    for card in detected_objects:
        class_name = card["class_name"]
        class_name_lower = class_name.lower()

        # Исключаем классы волн и 'available'
        if class_name_lower.startswith("w") or class_name_lower in ["available", "x4"]:
            continue

        # Проверяем, была ли карта временно понижена в приоритете
        if class_name_lower in failed_cards:
            failed_info = failed_cards[class_name_lower]
            last_failed_time = failed_info["last_failed_time"]
            failed_attempts = failed_info["attempts"]

            # Если прошло достаточно времени, восстанавливаем приоритет
            if current_time - last_failed_time > priority_reset_interval:
                obj_priorities[class_name] = initial_obj_priorities[class_name]
                del failed_cards[class_name_lower]
                logging.info(f"Приоритет карты {class_name} восстановлен.")
            elif failed_attempts >= max_failed_attempts:
                # Пропускаем карту, приоритет которой понижен
                logging.info(
                    f"Карта {class_name} временно исключена из выбора из-за неудачных попыток."
                )
                continue

        # Добавляем карту в список доступных
        cards.append(card)

    # Сортируем карты по текущему приоритету (наивысший первыми)
    sorted_cards = sorted(
        cards, key=lambda card: obj_priorities.get(card["class_name"], 0), reverse=True
    )
    return sorted_cards
