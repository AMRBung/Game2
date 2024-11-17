# inference_script.py

from yolo_model import yolo_model  # Импорт модели YOLO из файла yolo_model.py
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Путь к изображению
image_path = "screen.png"  # Замените на путь к вашему изображению

# Параметры инференса
CONFIDENCE_THRESHOLD = 0.5  # 50% уверенности
IOU_THRESHOLD = 0.3  # 30% порог перекрытия

# Выполнение инференса
try:
    result = yolo_model.predict(
        source=image_path, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD
    )
    predictions = result[0].boxes  # Получение боксов для первого изображения

    detected_objects = []
    if predictions:
        for pred in predictions:
            # Извлечение класса объекта
            cls_np = pred.cls.cpu().numpy()
            if cls_np.size == 0:
                logging.warning("Класс объекта пустой, пропуск.")
                continue
            class_id = int(cls_np[0])
            class_name = yolo_model.names[class_id]

            # Извлечение уверенности объекта
            conf_np = pred.conf.cpu().numpy()
            if conf_np.size == 0:
                logging.warning("Уверенность объекта пустая, пропуск.")
                continue
            confidence = float(conf_np[0])

            # Извлечение координат bounding box
            bbox_np = pred.xyxy.cpu().numpy()
            if bbox_np.size < 4:
                logging.error(
                    f"Bounding box имеет недостаточное количество координат: {bbox_np}"
                )
                continue
            bbox = bbox_np[0]  # Предполагается, что bbox_np имеет форму (1, 4)
            if len(bbox) < 4:
                logging.error(
                    f"Bounding box имеет недостаточное количество координат: {bbox}"
                )
                continue
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            detected_objects.append(
                {
                    "class_name": class_name,
                    "x": x_center,  # Центр по оси X
                    "y": y_center,  # Центр по оси Y
                    "width": width,
                    "height": height,
                    "confidence": confidence,
                }
            )
            logging.info(f"Обнаружен объект: {class_name}")
            logging.info(
                f"Координаты: x={x_center}, y={y_center}, ширина={width}, высота={height}"
            )
            logging.info(f"Уверенность: {confidence}\n")
    else:
        logging.info("Объекты не обнаружены на изображении.")

    # Дополнительная обработка, если необходимо
    # Например, сохранение результатов или дальнейшая логика

except Exception as e:
    logging.error(f"Ошибка при выполнении инференса: {e}")
