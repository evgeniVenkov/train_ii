import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

from torchvision.transform import v2
import torchvision
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import os
import json
import numpy as np

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
    def __getitem__(self,index):
        file_path, target = self.data_list[index]
        sample = np.array(Image.open(file_path))

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
def get_data():
    path_test = r'C:\Users\admin\PycharmProjects\my_train_ii\train_pytorch\mnist\testing'
    path_train = os.path.join(os.path.dirname(__file__), r'mnist\training')


    train_dataset = MnistDataset(path_train)
    test_dataset = MnistDataset(path_test)

    train_data, val_data = random_split(train_dataset, [0.8, 0.2])

    train_loader = DataLoader(train_data, batch_size=16,shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_Loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader, test_Loader

train_data,val_data,test_data = get_data()

# print(len(train_data))
# print(len(val_data))
# print(len(test_dataset))


#
# print(len(train_dataset), len(test_dataset))
#
# img, one_hot = train_dataset[35687]
#
# cls = train_dataset.classes[one_hot]
# print(f"Class: {cls}")
#
# plt.imshow(img, cmap='gray')
# plt.show()

# for name, idx in train_dataset.class_to_idx.items():
#     one_hot_vector = [(i == idx)*1 for i in range(10)]
#     print(name, one_hot_vector)

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

model = my_model(784,10)
loss_fn = nn.CrossEntropyLoss()
opt_class = torch.optim.Adam(model.parameters(), lr = 0.001)

input = torch.rand([16,784],dtype=torch.float32)
output = model(input)
print(output.shape)