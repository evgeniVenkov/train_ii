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
class MyDynamicModelWithStep(nn.Module):
    def __init__(self, input_channels=1, hidden_size=128, num_classes=6):
        super(MyDynamicModelWithStep, self).__init__()
        # Извлечение начальных признаков из входных изображений
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)  # Извлекает начальные признаки
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Извлекает более сложные признаки
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Глубокие признаки объектов

        # Рекуррентный слой для обработки последовательности
        self.rnn = nn.LSTM(64 * 51 * 80, hidden_size, batch_first=True)  # Сохраняет информацию о временной зависимости

        # Полносвязный слой для обработки шага
        self.fc_step = nn.Linear(1, 16)  # Вход для шага (индекса)

        # Полносвязный слой для классификации
        self.fc = nn.Linear(hidden_size + 16, num_classes)  # Учитывает скрытое состояние и шаг

    def forward(self, x, step):
        batch_size, seq_len, channels, height, width = x.size()

        # Обрабатываем каждый кадр через сверточные слои
        x = x.view(batch_size * seq_len, channels, height, width)  # Объединяем batch и sequence для обработки
        x = F.relu(self.conv1(x))  # Извлечение начальных признаков
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Уменьшение размерности в 2 раза
        x = F.relu(self.conv2(x))  # Извлечение более сложных признаков
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))  # Извлечение глубоких признаков объектов
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Преобразуем сверточный выход в вектор
        x = x.view(x.size(0), -1)  # (batch_size * seq_len, features)
        x = x.view(batch_size, seq_len, -1)  # Восстанавливаем последовательность

        # Пропускаем через рекуррентный слой
        _, (hidden, _) = self.rnn(x)  # Получаем скрытое состояние последнего слоя
        hidden = hidden[-1]  # Используем последнее скрытое состояние


        step = torch.tensor(step).unsqueeze(0).float()  # Преобразуем в тензор и добавляем размерность по оси 0
        step = step.to(device)
        step = step.unsqueeze(1)

        step = F.relu(self.fc_step(step))  # Обрабатываем шаг

        # Объединяем скрытое состояние и шаг
        combined = torch.cat((hidden, step), dim=1)

        # Пропускаем через полносвязный слой для предсказания
        out = self.fc(combined)  # Выходной результат, представляющий предсказание класса
        return out
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
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
def get_target(hot):
    index = torch.argmax(hot).item()
    zero_hot = torch.zeros_like(hot)
    print(hot)
    print("text")
    if index == 0:
        zero_hot[0, 0] = 0
        zero_hot[0, 1] = 1
        zero_hot[0, 2] = 1
        zero_hot[0, 3] = 1


    elif index == 1:
        zero_hot[0, 0] = 1
        zero_hot[0, 1] = 0
        zero_hot[0, 2] = 1
        zero_hot[0, 3] = 1

    elif index == 2:
        zero_hot[0, 0] = 1
        zero_hot[0, 1] = 1
        zero_hot[0, 2] = 0
        zero_hot[0, 3] = 1

    elif index == 3:
        zero_hot[0, 0] = 1
        zero_hot[0, 1] = 1
        zero_hot[0, 2] = 1
        zero_hot[0, 3] = 0
    else:
        zero_hot[0, 0] = 1

    return zero_hot.to(torch.float32)
def train_fail(model,out_buffer,history_img,loss_fn,optimizer,step):
    global list_loss
    model.train()

    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        param_group['lr'] = 0.01


    out = model(history_img,step)
    target = get_target(out_buffer[-1])

    loss = loss_fn(out, target)
    # adjust_learning_rate(optimizer,loss,0.01,0.0001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    list_loss.append(loss.item())
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return model
def train_sucess(model,out_buffer,history_img,loss_fn,optimizer,step):
    global list_loss
    model.train()
    for i in range(-5,0):
        if history_img.size(1) >= 5:
            continue
        old = torch.zeros_like(out_buffer[i])
        old[0,old.argmax()] = 1.4

        out = model(history_img[i],step+i)
        loss = loss_fn(out,old)
        # adjust_learning_rate(optimizer,loss,0.005,0.0001)

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

buffer_size = 15
image_buffer = deque(maxlen=buffer_size)  # Хранит скриншоты
out_buffer = deque(maxlen=buffer_size)

model = MyDynamicModelWithStep().to(device)
# load = torch.load("pet_model.pt")
# model.load_state_dict(load)

#штуки со скоростью
lr = 0.00000001
high_loss_threshold = 3.0
low_loss_threshold = 1.1


# Оптимизатор и функция потерь
loss_fn = nn.CrossEntropyLoss ()  # Для one-hot вектора
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Счетчики
score = None  # Очки
heals = 5
fine = None # Штраф
count = 0
step = 1.0

list_loss = []

#картинки
history_img_tens = None

for st in range(100000):
    model.train()

    img = get_scrin()  # скрин

    tens = image_to_tensor(img).to(device)
    tens = tens.unsqueeze(0)
    tens = tens.unsqueeze(0)
    step +=1.0

    if history_img_tens is None:
        history_img_tens = tens


    if history_img_tens.shape[1] >= 20:
        history_img_tens = torch.cat((history_img_tens[:, 1:], tens), dim=1)
    else:

        history_img_tens = torch.cat((history_img_tens, tens), dim=1)

    output = model(history_img_tens, step)# предсказание
    press_button_emulator(output)# нажатие кнопки
    event = get_event(img) # Получение события

    out_buffer.append(output)# сохранение предсказания
    target = torch.zeros_like(output)

    target[0, 0] = 0.3
    target[0,1] = 0.3
    target[0, 2] = 0.3
    target[0, 3] = 0.3
    target[0, 4] = 0.3
    target[0, 5] = 0.3

    if event[1]and step > 6:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            param_group['lr'] = 0.1
        heals -= 1
        print(f"[{step}] Неудача! Жизни {heals}")
        heals -= 1
        # model = train_fail(model, out_buffer, history_img_tens, loss_fn, optimizer, step)
        target = get_target(target)

        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        list_loss.append(loss.item())

        step = 0.0
        count +=1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        continue
    elif event[0]and step > 6:
        print(f"[{step}] Успех!")
        # model = train_sucess(model,out_buffer,history_img_tens,loss_fn,optimizer,step)
        count +=1




    train_out = model(history_img_tens, step)
    loss = loss_fn(train_out,target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    list_loss.append(loss.item())


    if count >= 5:
        plot_loss(list_loss)
        count = 0


