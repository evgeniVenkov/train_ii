import pygame
import random

class FlappyBirdGame:
    def __init__(self):
        pygame.init()
        self.screen_width = 400
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.gravity = 0.25
        self.bird_movement = 0
        self.bird = pygame.Rect(50, self.screen_height // 2, 34, 34)
        self.pipe_width = 60
        self.pipe_gap = 150
        self.reset()

    def reset(self):
        self.bird.y = self.screen_height // 2
        self.bird_movement = 0
        self.pipe_x = self.screen_width
        self.pipe_height = random.randint(200, 400)
        self.score = 0
        self.done = False

    def step(self, action):
        # Обработка действия
        if action == 1:  # Прыжок
            self.bird_movement = -7

        # Обновление состояния
        self.bird_movement += self.gravity
        self.bird.y += self.bird_movement
        self.pipe_x -= 3

        # Проверка столкновений
        if self.bird.y <= 0 or self.bird.y >= self.screen_height:
            self.done = True

        # Обновление трубок
        if self.pipe_x < -self.pipe_width:
            self.pipe_x = self.screen_width
            self.pipe_height = random.randint(200, 400)
            self.score += 1

        # Возвращаем скриншот, награду и флаг завершения
        reward = 1 if not self.done else -1
        state = self.get_state()
        return state, reward, self.done

    def render(self):
        self.screen.fill((255, 255, 255))
        pygame.draw.rect(self.screen, (0, 0, 255), self.bird)
        pygame.draw.rect(self.screen, (0, 255, 0), (self.pipe_x, 0, self.pipe_width, self.pipe_height))
        pygame.draw.rect(self.screen, (0, 255, 0), (self.pipe_x, self.pipe_height + self.pipe_gap, self.pipe_width, self.screen_height - self.pipe_height - self.pipe_gap))
        pygame.display.flip()

    def get_state(self):
        # Возвращаем состояние в виде скриншота
        return pygame.surfarray.array3d(self.screen)
