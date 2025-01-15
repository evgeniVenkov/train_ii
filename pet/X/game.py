import random

def print_game(board):

    game = []

    for i in board:
        if i == 0:
            game.append(" ")
        elif i == 1:
            game.append("X")
        else:
            game.append("O")

    print(f'{game[0]}|{game[1]}|{game[2]}\n'
          f'------\n'
          f'{game[3]}|{game[4]}|{game[5]}\n'
          f'------\n'
          f'{game[6]}|{game[7]}|{game[8]}')
def game_status(board):

    if board[0] == 1 and board[1] == 1 and board[2] == 1:
        return 1
    elif board[3] == 1 and board[4] == 1 and board[5] == 1:
        return 1
    elif board[6] == 1 and board[7] == 1 and board[8] == 1:
        return 1


    elif board[0] == 1 and board[3] == 1 and board[6] == 1:
        return 1
    elif board[1] == 1 and board[4] == 1 and board[7] == 1:
        return 1
    elif board[2] == 1 and board[5] == 1 and board[8] == 1:
        return 1

    elif board[1] == 1 and board[4] == 1 and board[8] == 1:
        return 1
    elif board[2] == 1 and board[4] == 1 and board[6] == 1:
        return 1



    if board[0] == -1 and board[1] == -1 and board[2] == -1:
        return -1
    elif board[3] == -1 and board[4] == -1 and board[5] == -1:
        return -1
    elif board[6] == -1 and board[7] == -1 and board[8] == -1:
        return -1


    elif board[0] == -1 and board[3] == -1 and board[6] == -1:
        return -1
    elif board[1] == -1 and board[4] == -1 and board[7] == -1:
        return -1
    elif board[2] == -1 and board[5] == -1 and board[8] == -1:
        return -1

    elif board[1] == -1 and board[4] == -1 and board[8] == -1:
        return -1
    elif board[2] == -1 and board[4] == -1 and board[6] == -1:
        return -1


    return 0
def random_bot(board):
    free_positions = [i for i, cell in enumerate(board) if cell == 0]

    if free_positions:
        move = random.choice(free_positions)
        board[move] = -1
    return board

board = [0,0,0,0,0,0,0,0,0]

def game_step():
    global board
    print_game(board)

    status = game_status(board)
    if status == 1:
        print_game(board)
        print("Viсtory!\n")
        board = [0,0,0,0,0,0,0,0,0]

    board = random_bot(board)
    status = game_status(board)
    if status == -1:
        print_game(board)
        print("loss!\n")
        board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    if board.count(0) == 0:
        print("Draw!")
        board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
