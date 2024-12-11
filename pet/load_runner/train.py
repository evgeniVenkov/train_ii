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
class Lode_Dataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.data_list = []
        self.targets = []
        TOTAL = 553
        path_loop = tqdm(total=TOTAL, desc='Loading data')
        for path_dir, dir_list, file_list in os.walk(self.path):
            if path_dir == self.path:
                self.classes = dir_list
                print(dir_list)
                self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
                continue

            cls = path_dir.split(os.sep)[-1]
            for name_file in file_list:
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

        self.data_list = torch.stack(self.data_list)  # Собираем список в один тензор
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        target = self.targets[idx]

        one_hot_target = F.one_hot(target, num_classes=len(self.classes)).float()
        return sample, one_hot_target

# Определяем параметры устройства и загрузчика данных
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Преобразуем в черно-белое изображение
    transforms.ToTensor(),  # Преобразуем в тензор
])


path = r"C:\Users\admin\PycharmProjects\train_ii\pet\load_runner\data"

data = Lode_Dataset(path, transform)
data_loader = DataLoader(data, batch_size=32, shuffle=True)


# Укажите размеры тренировочного и валидационного наборов
validation_split = 0.2  # 20% данных для валидации
dataset_size = len(data)
val_size = int(dataset_size * validation_split)
train_size = dataset_size - val_size

# Разделение на тренировочный и валидационный наборы
train_data, val_data = random_split(data, [train_size, val_size])

# Создаем загрузчики данных
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# model = MyModel().to(device)



epochs = 20
loss_list = []
loss_val = []

model = MyModel().to(device)
load = torch.load("pet_model.pt")
model.load_state_dict(load)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)


# model.eval()
# x,y = data[50]
# x = x.unsqueeze(0).to(device)
# x = model(x)
#
# print(x)
# print(y)
# exit()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    loop_data = tqdm(train_loader)
    loop_val = tqdm(val_loader)
    for x, target in loop_data:
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
        for x, target in loop_val:
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
plt.plot(loss_val, label="Validation Loss", linestyle="--")  # График потерь на валидации

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()  # Добавление легенды для обозначения линий
plt.show()











