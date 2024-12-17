import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchsummary import summary

class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,(3,3))
        #643  // 408
        self.act =nn.ReLU()

        self.linear = nn.Linear(10*643*408,500)
        self.linear2 = nn.Linear(500,100)
        self.linear3 = nn.Linear(100,6)

    def forward(self,x):
        x = self.act(self.conv1(x))
        x = torch.flatten(x,start_dim =1)
        x = self.act(self.linear(x))
        x = self.act(self.linear2(x))
        return self.linear3(x)


class Model1(nn.Module):
# колличество свёрток не решает пред проблемы
    def __init__(self):
        super().__init__()
        #(1,1,645,410)
        self.conv1 = nn.Conv2d(1, 10, (3, 3))#643 // 408
        self.conv2 = nn.Conv2d(10,20,(3,3))#641 // 406
        self.conv3 = nn.Conv2d(20,50,(3,3))#639.0 //404.0
        self.act = nn.ReLU()
        self.linear = nn.Linear(50 * 639 * 404, 6)  # Учтён размер после свёртки

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)# Свёртка
        x = torch.flatten(x, start_dim=1)  # Преобразуем в вектор
        x = self.linear(x)  # Линейный слой
        x = self.act(x)  # Активация
        return x
class Model(nn.Module):
    # очень простая модель
    # [[0.0776, 0.0000, 0.0000, 0.2459, 0.0715, 0.0220]]
    # если таргет попадлёт в нолевые то модель вообще не будет обучаться всё станет нолями
    # Total params: 15, 740, 746
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, (3, 3))
        self.act = nn.ReLU()
        self.linear = nn.Linear(10 * 643 * 408, 6)  # Учтён размер после свёртки

    def forward(self, x):
        x = self.conv1(x)  # Свёртка
        x = torch.flatten(x, start_dim=1)  # Преобразуем в вектор
        x = self.linear(x)  # Линейный слой
        x = self.act(x)  # Активация
        return x
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# Пример использования
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 0.001

# Загрузка и обработка изображения
img = Image.open('game_screenshot.png')
img = transform(img).to(device)  # Преобразование в тензор
img = img.unsqueeze(0)  # Добавляем batch размерности

model = Model2().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

summary(model,input_size = (1,645,410))

model.train()

# Обучение
while True:
    output = model(img)  # Выход модели
    print(f"Model output: {output}")

    try:
        num = int(input("Enter class index (0-5): "))  # Пользовательский ввод
        if num < 0 or num >= 6:
            print("Invalid index. Please enter a number between 0 and 5.")
            continue
    except ValueError:
        print("Invalid input. Please enter an integer.")
        continue

    target = torch.tensor([num], device=device)  # Целевая метка (скаляр)

    # Вычисление ошибки и оптимизация
    loss = loss_fn(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item()}")
