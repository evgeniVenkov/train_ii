import numpy as np
import matplotlib.pyplot as plt
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sparse_cross(z, y):
    return -np.log(z) if y == 1 else -np.log(1 - z)


def sig_der(t):
    sig = sigmoid(t)
    return sig * (1 - sig)


# Ваши данные
dataset = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0]]

# Инициализация переменных
w = np.random.uniform(-0.1, 0.1, (2, 1))
b = np.random.uniform(-0.1, 0.1, (1, 1))
loss_arr = []
ALPHA = 0.1  # Темп обучения
EPOCHS = 2000

for epoch in range(EPOCHS):
    random.shuffle(dataset)
    epoch_loss = 0  # Для усреднённой ошибки на эпоху
    print(f"\nEpoch: {epoch + 1}")

    for i, data in enumerate(dataset):
        x = np.array(data[:2]).reshape(1, -1)
        y = data[2]

        # Прямой проход
        t = x @ w + b
        z = sigmoid(t)
        e = sparse_cross(z, y)
        epoch_loss += e

        # Вывод значений для текущего шага
        print(f"  Data index: {i}")
        print(f"    x: {x}")
        print(f"    y: {y}")
        print(f"    t (input to sigmoid): {t}")
        print(f"    z (sigmoid output): {z}")
        print(f"    Loss: {e}")

        # Обратное распространение
        de_dz = z - y
        de_dt = de_dz * sig_der(t)
        de_dw = x.T @ de_dt
        de_db = de_dt

        # Вывод значений градиентов перед обновлением
        print(f"      de_dw: {de_dw}")
        print(f"      de_db: {de_db}")

        # Обновление весов и смещения
        w -= ALPHA * de_dw
        b -= ALPHA * de_db

        # Вывод весов и смещения после обновления
        print(f"    Updated weights: {w}")
        print(f"    Updated bias: {b}")

    # Добавляем усреднённую ошибку за эпоху
    avg_loss = epoch_loss / len(dataset)
    loss_arr.append(avg_loss.item())

    print(f"  Average loss for epoch {epoch + 1}: {avg_loss}")

# Построение графика ошибки
plt.plot(loss_arr)
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Training Loss over Epochs")
plt.show()
