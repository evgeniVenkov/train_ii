import os
import json


path = os.path.join(os.path.dirname(__file__),"dataset")
print(path)

list_name = os.listdir(path)


print("coords.json" in list_name)

with open(os.path.join(path, "coords.json"), "r") as file:
    cords_dir = json.load(file)

name = list_name[1]
path = os.path.join(path, name)


print(cords_dir[name])


from torch.utils.data import Dataset
from PIL import Image
class My_class(Dataset):
    def __init__(self, path, transform = None):
        self.path = path
        self.transform = transform

        self.file_names = os.listdir(path)
        if "coords.json" in self.file_names:
            self.file_names.remove("coords.json")

        self.len = len(self.file_names)

        with open(os.path.join(self.path,"coords.json"), 'r') as file:
            self.coords = json.load(file)

    def __len__(self):
        return self.len
    def __getitem__(self,index):
        file_name = self.file_names[index]
        coords = self.coords[file_name]

        img = Image.open(os.path.join(self.path,file_name))

        if self.transform is not None:
            img = self.transform(img)

        return img, coords


