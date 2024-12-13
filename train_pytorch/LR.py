import numpy as np
import random

# Создание среды (4x4)
class SimpleEnvironment:
    def __init__(self):
        self.state_space = 16  # 4x4 сетка
        self.action_space = 4  # Действия: вверх, вниз, влево, вправо
        self.goal_state = 15  # Целевая клетка (нижний правый угол)
        self.reset()

    def reset(self):
        self.agent_position = 0  # Начальная позиция (верхний левый угол)
        return self.agent_position

    def step(self, action):
        row, col = divmod(self.agent_position, 4)

        if action == 0 and row > 0:  # Вверх
            row -= 1
        elif action == 1 and row < 3:  # Вниз
            row += 1
        elif action == 2 and col > 0:  # Влево
            col -= 1
        elif action == 3 and col < 3:  # Вправо
            col += 1

        self.agent_position = row * 4 + col
        reward = 1 if self.agent_position == self.goal_state else -0.1
        done = self.agent_position == self.goal_state
        return self.agent_position, reward, done

# Q-обучение
class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0):
        self.q_table = np.zeros((state_space, action_space))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3)  # Случайное действие
        return np.argmax(self.q_table[state])  # Лучшее известное действие

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.max(self.q_table[next_state])
        td_target = reward + self.discount_factor * best_next_action
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

# Основной цикл обучения
env = SimpleEnvironment()
agent = QLearningAgent(state_space=env.state_space, action_space=env.action_space)

num_episodes = 500

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_table(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    agent.decay_exploration()
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Демонстрация после обучения
state = env.reset()
done = False
print("Trained Agent's Path:")

while not done:
    action = np.argmax(agent.q_table[state])
    state, _, done = env.step(action)
    print(f"State: {state}")
