import torch
import torchvision
from torchvision.transforms import v2
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

import os
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import json

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

        img = Image.open(path)
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


model = my_model(64*64,2)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

input = torch.rand([16,64*64])
output = model(input)
print(output.shape)


# path = os.path.join(os.path.dirname(__file__),"dataset")
#
# transform = v2.Compose([
#     v2.ToImage(),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Normalize(mean = (0.5,), std = (0.5,))
#
# ])
#
#
# sq_data =  squareDataset(path, transform)
#
# img, coord = sq_data[548]
# print(f"type: {type(img)}")
# print(f"shape: {img.shape}")
# print(f"Dt: {img.dtype}")
# print(f" min: {img.min()} max: {img.max()}")
# print("cls")
# print(f"coords: {type(coord)}")

# print(len(sq_data))
# img, cord = sq_data[130]
# print(cord)
# plt.scatter(cord[0],cord[1],marker='d',color='red')
# plt.imshow(img,cmap='gray')
# plt.show()

# train_set,val_set, test_set = random_split(sq_data,[0.7,0.1,0.2])
#
# train_loader = DataLoader(train_set,batch_size = 16, shuffle=True)
# val_loader = DataLoader(val_set, batch_size = 16, shuffle = False)
# test_loader = DataLoader(test_set, batch_size = 16, shuffle = False)

