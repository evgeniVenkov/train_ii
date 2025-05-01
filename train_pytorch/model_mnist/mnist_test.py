import torch
from torch.utils.data import Dataset
import os

class MnistDataset(Dataset):
    def __init__(self, path,transform):
        self.path = path
        self.transform = transform


    def __len__(self):
        pass
    def _getitem__(self,index):
        pass


path = "../mnist/testing"
classes = []

for path_dirs, dirs, files in os.walk(path):
    print(f"путь к файлу: {path_dirs}")
    print(f"вложенные папки: {dirs}")
    print(f"вложенные файлы: {files}")
    if path_dirs == path:
        classes = dirs
        continue

print(classes)