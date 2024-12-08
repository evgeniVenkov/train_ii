import subprocess
import time
import pygetwindow as gw
from PIL import Image
from torchvision import transforms
import pyautogui
import numpy as np
import torch.nn as nn
import torch


def start():
    # Путь к файлу игры и DOSBox
    dosbox_path = "C:\\Program Files (x86)\\DOSBox-0.74-3\\DOSBox.exe"
    game_path = "C:\\dendi_game\\lode\\LR.COM"

    # Запуск DOSBox с игрой
    subprocess.Popen([dosbox_path, game_path])

    # Подождать, пока игра загрузится
    time.sleep(3)

    # Получаем окно игры
    window = gw.getWindowsWithTitle('DOSBox')[0]
    window.moveTo(0, 0)  # Перемещаем окно в координаты (x, y)
    return window


# Функция для захвата изображения
def capture_game_screen(region=(0, 0, 645, 410)):
    screenshot = pyautogui.screenshot(region=region)
    return screenshot


# Функция для преобразования изображения в тензор с двумя каналами
def image_to_tensor(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Преобразуем в черно-белое изображение
        transforms.ToTensor(),  # Преобразуем в тензор
    ])
    image_tensor = transform(image)
    return image_tensor

class MyModel(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.linear = nn.Linear(inp, 500) #264 450
        self.act = nn.ReLU()
        self.linear1 = nn.Linear(500, 100)
        self.linear2 = nn.Linear(100, out)#6

    def forward(self, img,):
        x = self.linear(img)
        x = self.act(x)
        x = self.linear1(x)
        x = self.act(x)
        return self.linear2(x)
start()  # Запускаем игру
actions = ['no_action', 'left', 'right', 'dig', "up", "down"]


image = capture_game_screen()
image_tensor = image_to_tensor(image) #torch.Size([1, 410, 645])


# Преобразуем тензор обратно для проверки
image_array = image_tensor.squeeze(0).numpy()  # Убираем лишнюю ось
image_back = Image.fromarray((image_array * 255).astype(np.uint8), mode="L")

# Показываем изображение
image_back.show()

# Параметры для игры
time.sleep(2)  # Пауза для проверки



