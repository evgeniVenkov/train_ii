import subprocess
import time
import pygetwindow as gw
from PIL import Image

from matplotlib import pyplot as plt

from torchvision import transforms
import pyautogui
import numpy as np
import torch.nn as nn
import torch
from collections import deque
import torch.nn.functional as F



torch.autograd.set_detect_anomaly(True)
def press_button_emulator(one_hot_vector):
    button_index = torch.argmax(one_hot_vector).item()
    try:
        button = BUTTONS[button_index]
        print(f"Нажимаем кнопку: {button}")

        # Нажать кнопку через pyautogui
        pyautogui.keyDown(button)
        time.sleep(0.1)  # Удержание
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
def get_scrin(region=(0, 0, 645, 410)):
    screenshot = pyautogui.screenshot(region=region)
    return screenshot
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
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
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
    img_score = img.crop((100, 395, 240, 410))
    img_fine = img.crop((310, 395, 420, 410))

    return [get_score(img_score),get_fine(img_fine)]
def get_fine(img):
    global fine
    global heals

    if fine == None:
        fine = image_to_tensor(img)
        return False
    else:
        new_score = image_to_tensor(img)
        if not torch.equal(fine, new_score):
            fine = new_score
            return True
        return False
def get_score(img):
    global score
    if score == None:
        score = image_to_tensor(img)
        return False
    else:
        new_score = image_to_tensor(img)
        if not torch.equal(score, new_score):
            score = new_score
            return True
        return False
def restart():
    pass
def adjust_learning_rate(optimizer, loss, high_lr, low_lr):
    if loss > high_loss_threshold:
        for param_group in optimizer.param_groups:
            param_group['lr'] = high_lr
    elif loss < low_loss_threshold:
        for param_group in optimizer.param_groups:
            param_group['lr'] = low_lr
def get_target(hot):
    index = torch.argmax(hot).item()
    zero_hot = torch.zeros_like(hot)

    if index == 0:
        zero_hot[0,1] = 0.9
        zero_hot[0, 2] = 0.4
        zero_hot[0, 4] = 0.3
    elif index == 1:
        zero_hot[0,0] = 0.9
        zero_hot[0, 2] = 0.3
        zero_hot[0, 5] = 0.3
        zero_hot[0, 4] = 0.3
    elif index == 2:
        zero_hot[0,1] = 0.9
        zero_hot[0,0] = 0.9
    elif index == 3:
        zero_hot[0, 1] = 0.9
        zero_hot[0, 0] = 0.7
        zero_hot[0, 2] = 0.7

    return zero_hot.to(torch.float32)
def train_fail(model,out_buffer,image_buffer,loss_fn,optimizer):
    global list_loss
    model.train()
    for i, el in enumerate(out_buffer):
        out = model(image_buffer[i])
        target = get_target(el)
        loss = loss_fn(out, target)
        adjust_learning_rate(optimizer,loss,0.005,0.0001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        list_loss.append(loss.item())
    return model
def train_sucess(model,out_buffer,image_buffer,loss_fn,optimizer):
    global list_loss
    model.train()
    for i, el in enumerate(out_buffer):
        old = torch.zeros_like(el)
        old[0,el.argmax()] = 0.9
        out = model(image_buffer[i])
        loss = loss_fn(out,old)
        adjust_learning_rate(optimizer,loss,0.005,0.0001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        list_loss.append(loss.item())
    return model
def plot_loss(list_loss):

    if len(list_loss) > 10:
        list_loss = list_loss[int(len(list_loss) * 0.2):]
    plt.close('all')
    plt.figure(figsize=(10, 5))
    plt.plot(list_loss, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.title(f"{'Loss: '.join(f'{i:.3f}' for i in list_loss[-4:])}")
    plt.legend()
    plt.grid()
    plt.show()

window = start()  # Запускаем игру
BUTTONS = ['left','right','up', 'down','num7', 'num9']
device = "cuda" if torch.cuda.is_available() else "cpu"

buffer_size = 30
image_buffer = deque(maxlen=buffer_size)  # Хранит скриншоты
out_buffer = deque(maxlen=buffer_size)

model = MyModel().to(device)
load = torch.load("pet_model.pt")
model.load_state_dict(load)

#штуки со скоростью
lr = 0.001
high_loss_threshold = 2.0
low_loss_threshold = 1.1


# Оптимизатор и функция потерь
loss_fn = nn.CrossEntropyLoss ()  # Для one-hot вектора
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Счетчики
score = None  # Очки
heals = 5
fine = None # Штраф
count = 0
model.train()
list_loss = []

#картинки
previous_img = None
previous_target = None
previous_img_val = None
previous_target_val = None


for step in range(100000):
    model.eval()
    img = get_scrin()  # скрин
    tens = image_to_tensor(img).to(device)  # тензор
    if previous_img is None :
        previous_img = tens
    x = torch.cat((tens, previous_img),dim=1)


    output = model(x)# предсказание
    press_button_emulator(output)# нажатие кнопки
    event = get_event(img) # Получение события
    image_buffer.append(tens)# сохранение вектора
    out_buffer.append(output)# сохранение предсказания


    if event[1]:
        print(f"[{step}] Неудача! Жизни {heals}")
        heals -= 1
        model = train_fail(model,out_buffer,image_buffer,loss_fn,optimizer)
        image_buffer.clear()
        out_buffer.clear()
        count +=1

    elif event[0]:
        print(f"[{step}] Успех!")
        model = train_sucess(model,out_buffer,image_buffer,loss_fn,optimizer)

        count +=1

    if count >= 5:
        count = 0
        plot_loss(list_loss)










