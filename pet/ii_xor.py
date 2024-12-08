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
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Создаем модель
model = XORModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

num_epochs = 100
loss_arr = []


for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()


    output = model(X)
    loss = criterion(output, y)


    loss.backward()
    optimizer.step()


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
