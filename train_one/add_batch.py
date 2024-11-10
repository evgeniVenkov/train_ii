import numpy as np
import random
from sklearn import datasets
import matplotlib.pyplot as plt

# Загрузка данных
iris = datasets.load_iris()
dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]
loss_arr = []

# Параметры сети
input_dim = 4
output_dim = 3
h_dim = 10

# Инициализация весов
w1 = np.random.rand(input_dim, h_dim)
b1 = np.random.rand(1, h_dim)
w2 = np.random.rand(h_dim, output_dim)
b2 = np.random.rand(1, output_dim)


w1 = (w1 - 0.5) * 2 * np.sqrt(1/input_dim)
b1 = (b1 - 0.5) * 2 * np.sqrt(1/input_dim)
w2 = (w2 - 0.5) * 2 * np.sqrt(1/h_dim)
b2 = (b2 - 0.5) * 2 * np.sqrt(1/h_dim)


ALPHA = 0.001
NUM_EPOCHS = 500
BATCH_SIZE = 10

# Активационные функции и производные
def relu(x):
    return np.maximum(0, x)

def relu_der(t):
    return (t >= 0).astype(float)

def softmax_batch(t):
    e_x = np.exp(t - np.max(t, axis=1, keepdims=True))  # Численно стабильный softmax
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# Потери и функции преобразования
def sparse_cross_e_batch(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]) + 1e-9)  # Добавлен 1e-9 для стабильности

def to_full_batch(y, num_class):
    y_full = np.zeros((len(y), num_class))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full

# Функция для предсказания
def predict(x):
    t1 = x @ w1 + b1
    h1 = relu(t1)
    t2 = h1 @ w2 + b2
    z = softmax_batch(t2)
    return z

# Функция для вычисления точности
def calc_acc():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(dataset)
    return acc

# Цикл обучения
for ep in range(NUM_EPOCHS):
    random.shuffle(dataset)
    epoch_loss = 0
    for i in range(len(dataset) // BATCH_SIZE):
        batch_x, batch_y = zip(*dataset[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE])
        x = np.concatenate(batch_x, axis=0)
        y = np.array(batch_y)

        # Forward pass
        t1 = x @ w1 + b1
        h1 = relu(t1)
        t2 = h1 @ w2 + b2
        z = softmax_batch(t2)
        E = np.sum(sparse_cross_e_batch(z, y))
        epoch_loss += E

        # Backward pass
        y_full = to_full_batch(y, output_dim)
        de_dt2 = z - y_full
        de_dw2 = h1.T @ de_dt2
        de_db2 = np.sum(de_dt2, axis=0, keepdims=True)
        de_dh1 = de_dt2 @ w2.T
        de_dt1 = de_dh1 * relu_der(t1)
        de_dw1 = x.T @ de_dt1
        de_db1 = np.sum(de_dt1, axis=0, keepdims=True)

        # Update weights
        w1 -= ALPHA * de_dw1
        b1 -= ALPHA * de_db1
        w2 -= ALPHA * de_dw2
        b2 -= ALPHA * de_db2

    loss_arr.append(epoch_loss / (len(dataset) // BATCH_SIZE))

# Вывод точности и графика потерь
acc = calc_acc()
print("Точность:", acc)

plt.plot(loss_arr)
plt.xlabel("Эпоха")
plt.ylabel("Потери")
plt.title("График потерь")
plt.show()
