import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flag1, self.flag2, self.flag3, self.flag4 = True, True, True, True
        self.step1 = nn.Linear(9, 9)
        self.step2 = nn.Linear(9, 9)
        self.step3 = nn.Linear(9, 9)
        self.step4 = nn.Linear(9, 9)

    def forward(self, x):
        # Пропускаем через текущий активный слой
        if self.flag1:
            self.flag1 = False
            output = self.step1(x)
        elif self.flag2:
            self.flag2 = False
            output = self.step2(x)
        elif self.flag3:
            self.flag3 = False
            output = self.step3(x)
        elif self.flag4:
            self.flag4 = False
            output = self.step4(x)
        else:
            raise RuntimeError("Все флаги отключены. Сбросьте модель.")

        # Находим индекс максимального значения в свободных клетках (0)
        # Аргмаксим вернёт индекс лучшего хода
        return torch.argmax(output)

    def reset(self):
        self.flag1, self.flag2, self.flag3, self.flag4 = True, True, True, True
def make_move(model, board, criterion, optimizer):
    # Получаем индекс хода
    with torch.no_grad():
        move_index = model(board)

    # Проверяем, свободна ли клетка
    if board[move_index] != 0:
        # Если клетка занята, добавляем штраф
        loss = criterion(torch.tensor([1.0]), torch.tensor([0.0]))  # Неверный ход -> большая ошибка
        print("Неверный ход, штраф!")
    else:
        # Если клетка свободна, обновляем её
        board[move_index] = 1
        # Пример правильного хода: минимальная ошибка
        loss = criterion(torch.tensor([0.0]), torch.tensor([0.0]))

    # Обновляем параметры модели
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return board

# Инициализация
board = torch.tensor([0, 0, 0, 1, 0, -1, 0, 0, 0], dtype=torch.float32)
model = Model()
criterion = nn.MSELoss()  # Пример функции потерь
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Ходим и учим модель
board = make_move(model, board, criterion, optimizer)
print("Обновлённое поле:", board)
