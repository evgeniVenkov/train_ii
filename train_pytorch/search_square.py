import torch
import torchvision

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
        coords = self.dir_cords[name]

        if self.transform is not None:
            img = self.transform(img)

        return img, coords



path = os.path.join(os.path.dirname(__file__),"dataset")

sq_data =  squareDataset(path)

print(len(sq_data))

img, cord = sq_data[130]
print(cord)
plt.scatter(cord[0],cord[1],marker='d',color='red')
plt.imshow(img,cmap='gray')
plt.show()

