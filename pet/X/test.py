import random

# Пример игрового поля (3x3)
# 0 — пусто, 1 — игрок, 2 — противник
board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

# Функция для проверки победы
def check_win(board, player):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Горизонтали
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Вертикали
        [0, 4, 8], [2, 4, 6]              # Диагонали
    ]
    for condition in win_conditions:
        if all(board[i] == player for i in condition):
            return True
    return False

# Функция для оценки текущего состояния доски
def evaluate(board):
    if check_win(board, 1):
        return 10  # Победа игрока
    elif check_win(board, 2):
        return -10  # Победа противника
    return 0  # Ничья или не завершённая игра

# Минмакс с альфа-бета отсечением
def minimax(board, depth, is_maximizing, alpha, beta):
    score = evaluate(board)

    # Если игра завершена, возвращаем оценку
    if score == 10 or score == -10:
        return score

    # Если доска заполнена, то ничья
    if all(cell != 0 for cell in board):
        return 0

    if is_maximizing:
        best = -float('inf')
        for i in range(9):
            if board[i] == 0:  # Если клетка пустая
                board[i] = 2  # Противник делает ход
                best = max(best, minimax(board, depth + 1, not is_maximizing, alpha, beta))
                board[i] = 0  # Отменить ход
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
        return best
    else:
        best = float('inf')
        for i in range(9):
            if board[i] == 0:
                board[i] = 1  # Игрок делает ход
                best = min(best, minimax(board, depth + 1, not is_maximizing, alpha, beta))
                board[i] = 0  # Отменить ход
                beta = min(beta, best)
                if beta <= alpha:
                    break
        return best

# Функция для нахождения лучшего хода для противника
def find_best_move(board):
    best_val = -float('inf')
    best_move = -1

    for i in range(9):
        if board[i] == 0:
            board[i] = 2  # Противник делает ход
            move_val = minimax(board, 0, False, -float('inf'), float('inf'))
            board[i] = 0  # Отменить ход
            if move_val > best_val:
                best_move = i
                best_val = move_val

    return best_move

# Игрок и противник делают ходы по очереди
def play_game():
    global board
    while True:
        # Ход игрока
        player_move = int(input("Ваш ход (0-8): "))
        if board[player_move] == 0:
            board[player_move] = 1
        else:
            print("Неверный ход!")
            continue

        # Проверка на победу игрока
        if check_win(board, 1):
            print("Вы выиграли!")
            break

        # Проверка на ничью
        if all(cell != 0 for cell in board):
            print("Ничья!")
            break

        # Ход противника
        print("Ход противника...")
        move = find_best_move(board)
        board[move] = 2
        print(f"Противник сделал ход в клетку {move}")

        # Проверка на победу противника
        if check_win(board, 2):
            print("Противник выиграл!")
            break

        # Проверка на ничью
        if all(cell != 0 for cell in board):
            print("Ничья!")
            break

        # Отображаем доску
        print_board()

def print_board():
    for i in range(0, 9, 3):
        print(board[i:i+3])

# Запуск игры
play_game()
