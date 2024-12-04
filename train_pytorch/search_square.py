import torch

from torchvision.transforms import v2
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

import os
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import json
from tqdm import tqdm


class squareDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.list_names = os.listdir(path)
        if "coords.json" in self.list_names:
            self.list_names.remove('coords.json')

        self.len_data = len(self.list_names)
        with open(os.path.join(path,'coords.json'), 'r') as file:
            self.dir_cords = json.load(file)


    def __len__(self):
        return self.len_data

    def __getitem__(self,index):
        name = self.list_names[index]
        path = os.path.join(self.path, name)

        img = np.array(Image.open(path))


        coords = torch.tensor(self.dir_cords[name])

        if self.transform is not None:
            img = self.transform(img)

        return img, coords
class my_model(nn.Module):
    def __init__(self,input,output):
        super().__init__()
        self.layer1 = nn.Linear(input,128)
        self.layer2 = nn.Linear(128, output)
        self.act = nn.ReLU()

    def forward(self,x):
        x = self.layer1(x)
        x = self.act(x)
        out = self.layer2(x)
        return out
def get_data(path, batch_size = 32):
    teansform = v2.Compose([v2.ToImage(),
                            v2.ToDtype(torch.float32,scale = True),
                            v2.Normalize(mean = (0.5,), std = (0.5,))
                            ])

    data = squareDataset(path, teansform)

    train_size = int(0.72* len(data))
    val_size = int(0.18 * len(data))
    test_size = len(data) - train_size- val_size

    train_data, val_data, test_data = random_split(data, [train_size, val_size, test_size])


    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(test_data,batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def plot(LearningRate,run_train_loss,run_val_loss,accuracy_train,accuracy_val):
    plt.figure(figsize=(8, 5))
    plt.plot(run_train_loss[100:], label="Train Loss", color="blue")
    plt.plot(run_val_loss, label="Validation Loss", color="orange")
    plt.title(f"Training and Validation Loss learning {LearningRate}")  # Заголовок графика
    plt.xlabel("Iterations")  # Подпись оси X
    plt.ylabel("Loss")  # Подпись оси Y
    plt.legend()  # Легенда
    plt.grid()  # Включаем сетку для удобства
    plt.show()

    # График для accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(accuracy_train, label="Train Accuracy", color="blue")
    plt.plot(accuracy_val, label="Validation Accuracy", color="orange")
    plt.title(f"Training and Validation Accuracy Learning {LearningRate}")  # Заголовок графика
    plt.xlabel("Epochs")  # Подпись оси X
    plt.ylabel("Accuracy")  # Подпись оси Y
    plt.legend()  # Легенда
    plt.grid()  # Включаем сетку для удобства
    plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"
path = os.path.join(os.path.dirname(__file__),"dataset")




learning_rate = 0.001
EPOHS = 1
batch_size = 128

train_data, val_data,test_data = get_data(path, batch_size)
model = my_model(64*64,2).to(device)
loss_fn = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

val_loss = []
train_loss = []
train_acc = []
val_acc = []




for i in range(EPOHS):
    model.train()
    train_loop = tqdm(train_data, leave=False, desc=f"Epohs: {i+1}")
    epohs_loss = []
    correct = 0
    total = 0

    for x, target in train_loop:

        x = x.reshape(-1, 64 * 64).float().to(device)
        target = target.float().to(device)

        pred = model(x)

        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        epohs_loss.append(loss.item())

        mean_loss = np.mean(epohs_loss)

        threshold = 3.0

        correct += torch.sum(torch.abs(pred - target) < threshold).item()
        total += target.size(0)


        train_acc.append(correct / total)

        train_loop.set_description(f"train Epoch {i+1}"
                                   f" loss: {mean_loss:.4f}"
                                   f" accuracy: {(correct/(total*2)):.4f}")


model.eval()
val_loop = tqdm(val_data, leave=True)
epohs_loss = []
correct = 0
total = 0

with torch.no_grad():
    for x, target in val_loop:
        x = x.reshape(-1, 64 * 64).float().to(device)
        target = target.float().to(device)

        pred = model(x)
        loss = loss_fn(pred, target)

        epohs_loss.append(loss.item())
        val_loss.append(loss.item())
        mean_loss = np.mean(epohs_loss)

        total += target.shape[0]

        threshold = 3.0

        correct += (torch.abs(pred - target) < threshold).sum().item()


        val_acc.append(correct / total)

        val_loop.set_description(f"val Epoch {i+1} "
                                 f"loss: {mean_loss:.1f} "
                                 f"accuracy: {correct/total:.1f} ")



plot(learning_rate,train_loss,val_loss,train_acc,val_acc)

