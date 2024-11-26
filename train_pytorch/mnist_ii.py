import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

# from torchvision.transform import v2
import torchvision
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import os
import json
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class MnistDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.len_dataset = 0
        self.data_list = []

        for path_dir, dir_list, file_list in os.walk(path):
            if path_dir == path:
                self.classes = dir_list
                self.class_to_idx = {
                    cls_name: i for i, cls_name in enumerate(self.classes)

                }
                continue
            cls = path_dir.split('\\')[-1]

            for name_file in file_list:
                file_path = os.path.join(path_dir, name_file)
                self.data_list.append((file_path, self.class_to_idx[cls]))
            self.len_dataset += len(file_list)
    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        file_path, target = self.data_list[index]
        sample = np.array(Image.open(file_path).resize((28, 28)))
        sample = torch.tensor(sample, dtype=torch.float32)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def get_data():
    path_test = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'mnist\testing')
    path_train = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'mnist\training')


    train_dataset = MnistDataset(path_train)
    test_dataset = MnistDataset(path_test)

    train_data, val_data = random_split(train_dataset, [0.8, 0.2])

    train_loader = DataLoader(train_data, batch_size=16,shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_Loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader, test_Loader



class my_model(nn.Module):
    def __init__(self,input,output):
        super().__init__()
        self.layer1 = nn.Linear(input,128)
        self.layer2 = nn.Linear(128,output)
        self.act = nn.ReLU()

    def forward(self,x):
        x = self.layer1(x)
        x = self.act(x)
        out = self.layer2(x)

        return out

train_data,val_data,test_data = get_data()

print(len(train_data))
print(len(val_data))
print(len(test_data))


model = my_model(784,10).to(device)


loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr = 0.001)
EPOHS = 5

run_train_loss = []
mean_train_loss =[]
run_val_loss = []
mean_val_loss = []




for i in range(EPOHS):

    model.train()
    train_loop = tqdm(train_data,leave = False)
    for x, target in train_loop:
        x = x.reshape(-1,28*28).to(device)


        target = target.reshape(-1).to(torch.int64)


        pred = model(x)
        loss = loss_fn(pred,target)

        opt.zero_grad()
        loss.backward()

        opt.step()

        run_train_loss.append(loss.item())
        mean_train_loss = sum(run_train_loss)/ len(run_train_loss)


        train_loop.set_description(f"Epohs{i + 1} train loss {mean_train_loss:.4f}")

    model.eval()
    val_loop = tqdm(val_data,leave = False)
    with torch.no_grad():
        for x, target in val_loop:
            x = x.reshape(-1,28*28).to(device)

            target = target.reshape(-1).to(torch.int32)
            target = torch.eye(10)[target].to(device)

            pred = model(x)
            loss = loss_fn(pred,target)

            run_val_loss.append(loss.item())
            mean_val_loss = sum(run_val_loss)/len(run_val_loss)
            val_loop.set_description(f" epohs {i +1} mean val loss {mean_val_loss:.4f}")



input = torch.rand([16,784],dtype=torch.float32)
output = model(input)
print(output.shape)