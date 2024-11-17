# adb_screenshot_capture.py

import subprocess
import logging


def get_device_id():
    result = subprocess.run(["adb", "devices"], stdout=subprocess.PIPE, text=True)
    output = result.stdout
    lines = output.strip().split("\n")
    devices = [line.split("\t")[0] for line in lines if "\tdevice" in line]
    if not devices:
        raise RuntimeError("Не найдено подключенных устройств ADB.")
    # Если подключено несколько устройств, выберите нужное
    device_id = devices[0]  # Или укажите конкретный идентификатор
    return device_id


def capture_screenshot(device_id):
    try:
        with open("screen.png", "wb") as f:
            subprocess.run(
                ["adb", "-s", device_id, "exec-out", "screencap", "-p"], stdout=f
            )
        logging.info("Скриншот снят и сохранен как screen.png")
    except Exception as e:
        logging.error(f"Ошибка при снятии скриншота: {e}")


# Пример использования
if __name__ == "__main__":
    device_id = get_device_id()
    capture_screenshot(device_id)
