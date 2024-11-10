import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Входные данные
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]], dtype=torch.float32)

# Метки (выходные данные)
y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]], dtype=torch.float32)

# Определение модели
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # Первый слой: 2 входа, 5 нейронов в скрытом слое
        self.fc2 = nn.Linear(10, 1)   # Второй слой: 5 входов, 1 выход

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Применяем ReLU к первому слою
        x = torch.sigmoid(self.fc2(x))  # Применяем сигмоиду к выходному слою
        return x

# Создаем модель
model = XORModel()
criterion = nn.BCELoss()  # Функция потерь: бинарная кросс-энтропия
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Оптимизатор: Adam

num_epochs = 100
loss_arr = []

# Обучение модели
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()  # Обнуляем градиенты

    # Прямой проход
    output = model(X)
    loss = criterion(output, y)  # Вычисляем потерю

    # Обратный проход и оптимизация
    loss.backward()
    optimizer.step()

    # Сохраняем потерю
    loss_arr.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Оценка модели
model.eval()  # Устанавливаем модель в режим оценки
with torch.no_grad():  # Отключаем вычисление градиентов
    predictions = model(X)
    predicted_classes = (predictions > 0.5).float()  # Преобразуем предсказания в классы

# Вывод результатов
for i in range(len(X)):
    print(f"Input: {X[i].numpy()}, True Output: {y[i].numpy()}, Predicted Output: {predictions[i].numpy()}")

# Построение графика потерь
plt.plot(loss_arr)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.show()
