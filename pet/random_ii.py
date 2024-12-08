import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# Функция для генерации датасета
def generate_dataset(num_samples=1000, sequence_length=100, num_features=10, num_range=36):
    data = []    # Список для хранения входных данных
    targets = [] # Список для хранения целевых значений (101-е число)

    for _ in range(num_samples):
        # Генерируем случайные числа для создания последовательности с формой (sequence_length, num_features)
        sequence = np.random.randint(1, num_range + 1, (sequence_length, num_features))
        data.append(sequence)  # Добавляем последовательность с 10 признаками
        targets.append(np.random.randint(1, num_range + 1))  # Случайное целевое значение

    # Преобразуем списки в тензоры PyTorch и изменяем их форму
    data = torch.tensor(np.array(data), dtype=torch.float32)  # data будет иметь форму (num_samples, sequence_length, num_features)
    targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)  # Целевое значение в виде тензора

    return data, targets  # Возвращаем данные и цели
# Определение модели RNN
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()  # Инициализация базового класса
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # Создание слоя RNN
        self.fc = nn.Linear(hidden_size, output_size)  # Полносвязный слой для выхода

    def forward(self, x):
        out, _ = self.rnn(x)  # Пропускаем вход через слой RNN
        out = self.fc(out[:, -1, :])  # Используем последнее скрытое состояние
        return out  # Возвращаем выходные данные

# Проверяем, доступен ли GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Генерация данных
data, targets = generate_dataset(num_samples=500, num_features=5)  # Изменяем параметр num_features
data, targets = data.to(device), targets.to(device)  # Переносим данные на GPU

# Параметры модели
input_size = 5  # Размер входного слоя (количество признаков)
hidden_size = 64  # Размер скрытого слоя
output_size = 1   # Размер выходного слоя (предсказание одного числа)

# Создаем экземпляр модели и переносим её на GPU
model = SimpleRNN(input_size, hidden_size, output_size).to(device)

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()  # Используем среднеквадратичную ошибку как функцию потерь
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Используем Adam для оптимизации

loss_values = []


# Обучение модели
epochs = 50  # Количество эпох
batch_size = 10

for epoch in range(epochs):
    model.train()  # Переключаем модель в режим обучения
    epoch_loss = 0.0  # Переменная для суммирования лосса за эпоху


    for i in range(0, data.size(0), batch_size):
        model.train()  # Переключаем модель в режим обучения
        optimizer.zero_grad()  # Обнуляем градиенты оптимизатора
        outputs = model(data)  # Пропускаем данные через модель
        loss = criterion(outputs, targets)  # Вычисляем функцию потерь
        loss.backward()  # Вычисляем градиенты
        optimizer.step()  # Обновляем параметры модели
        epoch_loss += loss.item() # Суммируем лосс

        # Рассчитываем средний лосс за эпоху
    avg_loss = epoch_loss / (data.size(0) / batch_size)
    loss_values.append(avg_loss)

    # Выводим информацию о потере каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

sequence_length = data.shape[1]

# Пример прогнозирования нескольких следующих чисел после последовательности
num_predictions = 5  # Количество предсказаний
with torch.no_grad():  # Отключаем градиенты для тестирования
    for i in range(num_predictions):
        # Берём последовательности из конца обучающего набора
        test_seq = data[-(num_predictions - i)].view(1, sequence_length, input_size).to(device)
        predicted = model(test_seq)  # Делаем предсказание
        actual = targets[-(num_predictions - i)].item()  # Получаем фактическое значение
        # Выводим предсказанное и фактическое значение
        print(f'Следующее предсказанное число: {predicted.item()}, фактическое число: {actual}')



plt.plot(loss_values, label='Average Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Average Training Loss over Epochs')
plt.legend()
plt.show()