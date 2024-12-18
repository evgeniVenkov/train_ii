from game import FlappyBirdGame
import pygame
import torch
import torch.nn as nn
from torchvision import transforms
from collections import deque


class GameModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), bias=False),  # 32, 398, 598
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2), (2, 2), bias=False),  # 64, 199, 299
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3), bias=False),  # 32, 197, 297
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, (1, 1), (2, 2), bias=False),  # 3, 99, 149
            nn.BatchNorm2d(3),
            nn.MaxPool2d((3, 3))  # 3, 33, 48
        )
        self.linear = nn.Sequential(
            nn.Linear(3 * 33 * 49, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x


# Инициализация
size_img = 10
list_img = deque(maxlen=size_img)
list_target = deque(maxlen=size_img)
list_rewards = deque(maxlen=size_img)

model = GameModel()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
list_loss = []

transform = transforms.Compose([transforms.ToTensor()])

game = FlappyBirdGame()
running = True

# Накопление опыта
states = []
actions = []
rewards = []

while running:
    screen = game.get_state()
    tensor = transform(screen)
    tensor = tensor.unsqueeze(0)

    # Получаем выход модели
    out = model(tensor)
    action = torch.argmax(out).item()  # Агент выбирает действие (0 или 1)

    # Добавляем опыт
    states.append(tensor)
    actions.append(action)


    if action == 1:  # Прыжок
        _, _, done = game.step(1)
    else:  # Бездействие
        _, _, done = game.step(0)

    # Получаем вознаграждение
    reward = 1 if done else 0  # Можно использовать другую логику для вознаграждения
    rewards.append(reward)

    # Отображение игры
    game.render()

    # Если игра закончена, обучаем модель
    if done:
        model.train()

        # Обновление модели на основе накопленного опыта
        for i in range(len(states)):
            # Целевая метка (предсказание для выбранного действия)
            target = torch.tensor([actions[i]])

            # Прогноз модели
            out = model(states[i])

            # Потери
            loss = loss_fn(out, target)
            list_loss.append(loss.item())

            # Обратное распространение
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'loss: {loss.item():.4f}')

        print(f"Игра окончена. Очки: {game.score}")
        game.reset()  # Сброс игры для новой попытки

    # Ограничиваем FPS
    game.clock.tick(30)  # Ограничение FPS

pygame.quit()
