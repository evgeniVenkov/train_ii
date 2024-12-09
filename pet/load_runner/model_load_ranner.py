import subprocess
import time


import pygetwindow as gw
from PIL import Image

from torchvision import transforms
import pyautogui
import numpy as np
import torch.nn as nn
import torch

import torch.nn.functional as F
def press_button_emulator(one_hot_vector):
    """
    Эмулирует нажатие кнопки для эмулятора (DOSBox), используя one-hot вектор.

    :param one_hot_vector: list[int] - One-hot вектор с размером, равным количеству кнопок
    """
    button_index = torch.argmax(one_hot_vector).item()


    try:
        button = BUTTONS[button_index]
        print(f"Нажимаем кнопку: {button}")

        # Нажать кнопку через pyautogui
        pyautogui.keyDown(button)
        time.sleep(0.25)  # Удержание
        pyautogui.keyUp(button)

    except:
        print("Ошибка в нажатии кнопок")
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
def get_scrin(region=(0, 0, 645, 410)):
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
    def __init__(self, num_classes=6):
        super().__init__()
        # Сверточные слои
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Полносвязные слои
        self.fc1 = nn.Linear(64 * 51 * 80, 128)  # Размер зависит от выхода сверточной части
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Сверточная часть
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Уменьшение размерности в 2 раза
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Преобразуем в вектор
        x = x.view(x.size(0), -1)  # Разворачиваем тензор
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Без активации, так как softmax будет применяться в функции потерь
        return x
def show_img(img):
    img = img.squeeze(0).numpy()
    img = Image.fromarray((img * 255).astype(np.uint8),mode = 'L')
    img.show()
def get_event(img):
    imd_score = img.crop((100, 395, 240, 410))
    return get_score(imd_score)

def get_score(img):
    global score
    img.show()
    if score == None:
        score = image_to_tensor(img)
        return False
    else:
        new_score = image_to_tensor(img)
        if not torch.equal(score, new_score):
            score = new_score
            return True
        return False



heals = 7
score = 0

window = start()  # Запускаем игру
BUTTONS = ['up', 'down', 'left', 'right', 'z', 'x']
device = "cuda" if torch.cuda.is_available() else "cpu"


model = MyModel().to(device)

# Оптимизатор и функция потерь
criterion = nn.CrossEntropyLoss ()  # Для one-hot вектора
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Счетчики
score = None  # Очки
penalty = 0  # Штраф
model.train()
# Обучающий цикл
for step in range(1000):  # Задаем число итераций

    img = get_scrin()
    tens = image_to_tensor(img).to(device)

    tens = torch.reshape(tens, (1,) + tens.shape)


    # Прямой проход
    output = model(tens)

    press_button_emulator(output)
    event = get_event(img) # Получение события

    if event:
        score += 1
        print(f"[{step}] Успех! Предсказано: {output}")
        loss = criterion(output, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Обучение на последнем действии. Loss: {loss.item():.4f}")


    # else:
    #     penalty += 1
    #     print(f"[{step}] Штраф! Предсказано: {output}, Штраф: {penalty}")

        # Таргет известен только в момент штрафа



        # loss = criterion(output, target)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()





# написать окно вида от лица нейронки






