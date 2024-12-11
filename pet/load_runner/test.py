import os
import subprocess
import pygetwindow as gw
import pyautogui
from time import sleep, time


def start():
    # Путь к файлу игры и DOSBox
    dosbox_path = "C:\\Program Files (x86)\\DOSBox-0.74-3\\DOSBox.exe"
    game_path = "C:\\dendi_game\\lode\\LR.COM"

    # Запуск DOSBox с игрой
    subprocess.Popen([dosbox_path, game_path])

    # Подождать, пока игра загрузится
    sleep(3)

    # Получаем окно игры
    window = gw.getWindowsWithTitle('DOSBox')[0]
    window.moveTo(0, 0)  # Перемещаем окно в координаты (x, y)
    return window
# Параметры
save_folder = "screenshots"  # Папка для сохранения
region = (0, 0, 645, 410)  # Координаты области (x, y, ширина, высота)
screenshot_interval = 0.1  # Интервал между скриншотами в секундах
num_screenshots = 500  # Количество скриншотов

# Создаём папку, если её нет
os.makedirs(save_folder, exist_ok=True)
start()
# Цикл сохранения скриншотов
for i in range(1, num_screenshots + 1):
    screenshot = pyautogui.screenshot(region=region)  # Снимаем скриншот
    screenshot.save(os.path.join(save_folder, f"0.{i}.png"))  # Сохраняем в файл
    print(f"Скриншот {i} сохранён.")
    sleep(screenshot_interval)  # Пауза перед следующим скриншотом

print("Скриншоты успешно сохранены!")
