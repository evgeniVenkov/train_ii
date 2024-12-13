import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import re

class MyDynamicModelWithStep(nn.Module):
    def __init__(self, input_channels=3, hidden_size=128, num_classes=6):
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

        # Пропускаем шаг через полносвязный слой
        step = step.unsqueeze(1).float()  # Преобразуем шаг в форму [batch_size, 1]
        step = F.relu(self.fc_step(step))  # Обрабатываем шаг

        # Объединяем скрытое состояние и шаг
        combined = torch.cat((hidden, step), dim=1)

        # Пропускаем через полносвязный слой для предсказания
        out = self.fc(combined)  # Выходной результат, представляющий предсказание класса
        return out

class SequentialDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.data_list = []
        self.targets = []

        TOTAL = 118  # Количество изображений для отображения процесса загрузки
        path_loop = tqdm(total=TOTAL, desc='Loading data')

        for path_dir, dir_list, file_list in os.walk(self.path):
            if path_dir == self.path:
                self.classes = dir_list  # Классы (папки с изображениями)
                print(dir_list)
                self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
                continue

            cls = path_dir.split(os.sep)[-1]
            for name_file in sorted(file_list):  # Сортируем файлы для последовательности
                file_path = os.path.join(path_dir, name_file)
                image = Image.open(file_path).resize((645, 410))
                if self.transform:
                    image = self.transform(image)
                else:
                    image = transforms.ToTensor()(image)
                self.data_list.append(image)
                self.targets.append(self.class_to_idx[cls])
                path_loop.update()

        path_loop.close()

        # Перемещаем первые 48 фотографий в конец списка
        self.data_list = torch.stack(self.data_list)  # Преобразуем в один тензор
        self.targets = torch.tensor(self.targets, dtype=torch.long)

        self.data_list = torch.cat((self.data_list[48:], self.data_list[:48]))  # Переносим первые 48 в конец
        self.targets = torch.cat((self.targets[48:], self.targets[:48]))  # Переносим метки

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        target = self.targets[idx]
        return sample, target


class val_dataset(Dataset):
    def __init__(self, ):
        self.data = []
        self.targets = []

    def add_data(self, data, target):
        self.data.append(data)
        self.targets.append(target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# Определяем параметры устройства и загрузчика данных
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Преобразуем в черно-белое изображение
    transforms.ToTensor(),  # Преобразуем в тензор
])



path = r"C:\Users\admin\PycharmProjects\train_ii\pet\load_runner\data"

data = SequentialDataset(path, transform)
data_val = val_dataset()
data_train = val_dataset()

for i ,(img,target) in enumerate(data):
    if i % 3 == 0:
        data_val.add_data(img, target)
        continue
    data_train.add_data(img, target)

print(len(data_val))
print(len(data_train))
exit()



data_loader = DataLoader(data, batch_size=1, shuffle=False)


# Укажите размеры тренировочного и валидационного наборов
validation_split = 0.2  # 20% данных для валидации
dataset_size = len(data)
val_size = int(dataset_size * validation_split)
train_size = dataset_size - val_size

# Разделение на тренировочный и валидационный наборы
train_data, val_data = random_split(data, [train_size, val_size])

# Создаем загрузчики данных
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

# model = MyModel().to(device)



epochs = 20
loss_list = []
loss_val = []

model = MyModel().to(device)
# load = torch.load("pet_model.pt")
# model.load_state_dict(load)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# model.eval()
# x,y = data[50]
# x = x.unsqueeze(0).to(device)
# x = model(x)
#
# print(x)
# print(y)
# exit()

previous_img = None
previous_target = None
previous_img_val = None
previous_target_val = None
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    loop_data = tqdm(train_loader)
    loop_val = tqdm(val_loader,leave=False)
    for img, target in loop_data:
        if previous_img is None and previous_target is None:
            previous_img, previous_target = img, target

        x = torch.cat((img, previous_img), dim=1)
        previous_img, previous_target = img, target

        x = x.to(device)
        target = target.to(device)


        output = model(x)


        loss = loss_fn(output, target)
        epoch_loss += loss.item()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop_data.set_description(f"Epoch {epoch+1}/{epochs}; Loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        for img, target in loop_val:
            if previous_img_val is None and previous_target_val is None:
                previous_img_val, previous_target_val = img, target

            x = torch.cat((img, previous_img), dim=1)
            previous_img, previous_target = img, target


            x = x.to(device)
            target = target.to(device)
            output = model(x)
            loss = loss_fn(output, target)
            loss_val.append(loss.item())
            loop_val.set_description(f"Validation Loss {loss.item():.4f}")

    avg_loss = epoch_loss / len(data_loader)
    loss_list.append(avg_loss)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


torch.save(model.state_dict(), "pet_model.pt")

plt.plot(loss_list, label="Training Loss")  # График потерь на обучении
plt.plot(loss_val, label="Validation Loss", linestyle="-")  # График потерь на валидации

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()  # Добавление легенды для обозначения линий
plt.show()











