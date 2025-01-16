import torch
import torch.nn as nn
import torch.optim as optim
import game
import matplotlib.pyplot as plt


torch.autograd.set_detect_anomaly(True)

Co_fine = 0
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flag1, self.flag2, self.flag3, self.flag4,self.flag5 = True, True, True, True,True
        self.step1 = nn.Linear(9, 9)

        self.step2 = nn.Linear(9, 9)

        self.step3 = nn.Linear(9, 9)

        self.step4 = nn.Linear(9, 9)

        self.step5 = nn.Linear(9, 9)

    def forward(self, x):

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
        elif self.flag5:
            self.flag5 = False
            output = self.step5(x)
        else:
            raise RuntimeError("Все флаги отключены. Сбросьте модель.")


        return output

    def reset(self):
        self.flag1, self.flag2, self.flag3, self.flag4, self.flag5 = True, True, True, True,True
def make_move(model, board, criterion, optimizer):
    fine =  False
    global Co_fine

    if model.flag1:
        current_layer = model.step1
    elif model.flag2:
        current_layer = model.step2
    elif model.flag3:
        current_layer = model.step3
    elif model.flag4:
        current_layer = model.step4
    elif model.flag5:
        current_layer = model.step5

    # if board.tolist().count(0) == 1:
    #     board[board.index(0)] = 1
    #     return board, out
    out = model(board)

    move_index = out.argmax().item()
    print(out)
    print(move_index)


    if board[move_index] != 0:
        last_move = False
        print("Неверный ход, штраф!")
        Co_fine +=1
        fine = True
        if not model.flag5:
            model.flag5 = True
            last_move = True
        elif not model.flag4:
            model.flag4 = True
        elif not model.flag3:
             model.flag3 = True
        elif not model.flag2:
            model.flag2 = True
        elif not model.flag1:
            model.flag1 = True

        list_board = board.tolist()
        target = out.clone()
        target[move_index] = -1
        free_positions = [i for i, cell in enumerate(list_board) if cell == 0]

        for i in free_positions:
            target[i] = 1

        loss = criterion(out, target)
    else:
        #Верный ход
        board = board.clone()
        board[move_index] = 1
        target = out.clone()
        target[move_index] = 2
        loss = criterion(out, target)

    #Замораживаем параметры
    for param in model.parameters():
        param.requires_grad = False
    for param in current_layer.parameters():
        param.requires_grad = True


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Размораживаем параметры
    for param in model.parameters():
        param.requires_grad = True

    if fine:
        board,out = make_move(model, board, criterion, optimizer)


    return board, out

def train_model(criterion, optimizer,out):
    target = out.clone()
    target[out.argmax().item()] = -1
    loss = criterion(out, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

board = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)

model = Model()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

count_fine = []

win = 0
looser =0
draw = 0

epohs = 0

while True:
    board = torch.tensor(board, dtype=torch.float32)
    board, out = make_move(model, board, criterion, optimizer)

    board, status = game.game_step(board.tolist())
    if status != 0:
        epohs +=1
        count_fine.append(Co_fine)
        Co_fine = 0
        model.reset()
        if status == -1 or status == 2:
            # train_model(criterion, optimizer,out)
            if status == -1:
                looser += 1
            else:
                draw += 1
        else:
            win+=1
    # if epohs == 500:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.1
    # if epohs == 1000:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.05
    # if epohs == 1500:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.02
    # if epohs == 1800:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.009
    if epohs >= 1000:
        break

print(f'win : {win}\n'
      f'looser: {looser}\n'
      f'draw: {draw}')

plt.plot(count_fine)

plt.show()
plt.plot(count_fine[800:])

plt.show()
torch.save(model.state_dict(), 'model.pt')