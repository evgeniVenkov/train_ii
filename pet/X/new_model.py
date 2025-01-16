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
        self.step1 = nn.Linear(9, 16)
        self.step2 = nn.Linear(32, 64)
        self.step3 = nn.Linear(64, 32)
        self.step4 = nn.Linear(32, 16)
        self.step5 = nn.Linear(16, 9)

        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.step1(x))
        x = self.act(self.step2(x))
        x = self.act(self.step3(x))
        x = self.act(self.step4(x))
        x = self.act(self.step5(x))
        return x





def make_move(model, board, criterion, optimizer):
    fine =  False
    global Co_fine


    out = model(board)

    move_index = int(out.item())
    print(out)
    print(move_index)


    if board[move_index] != 0:
        last_move = False
        print("Неверный ход, штраф!")
        Co_fine +=1
        fine = True


        list_board = board.tolist()
        target = out.clone()
        target[move_index] = -1
        free_positions = [i for i, cell in enumerate(list_board) if cell == 0]

        for i in free_positions:
            target[i] = 1

        loss = criterion(out, target)
    else:
        board = board.clone()
        board[move_index] = 1
        loss = criterion(out, out)  # Правильный ход -> минимальная ошибка


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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
optimizer = optim.SGD(model.parameters(), lr=0.3)

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
    if epohs == 300:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.2
    if epohs == 600:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1
    if epohs >= 1000:
        break

print(f'win : {win}\n'
      f'looser: {looser}\n'
      f'draw: {draw}')

plt.plot(count_fine)

plt.show()
plt.plot(count_fine[850:])

plt.show()
torch.save(model.state_dict(), 'model.pt')