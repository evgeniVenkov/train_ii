import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import numpy as np
from PIL import Image, ImageDraw
import json
from torchvision.transforms import v2
from torchvision.transforms import functional as F

class sqDataset(Dataset):
    def __init__(self,path, transform=None):
        self.path = path
        self.transform = transform

        self.list_files = os.listdir(path)
        if "coords.json" in self.list_files:
            self.list_files.remove("coords.json")
        self.len = len(self.list_files)

        with open(os.path.join(self.path,"coords.json")) as f:
            self.coords = json.load(f)
    def __len__(self):
        return self.len

    def __getitem__(self,index):
        name = self.list_files[index]
        coords = torch.tensor(self.coords[name])
        path = os.path.join(self.path,name)
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)
        return image, coords
class model_square(nn.Module):
    def __init__(self,inp, out):
        super().__init__()
        self.layer1 = nn.Linear(inp,128)
        self.layer2 = nn.Linear(128,out)
        self.act = nn.ReLU()

    def forward(self,x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        return x

def draw_points(image, pred_point):

    image = image.convert("RGB")

    draw = ImageDraw.Draw(image)
    pred_x, pred_y = pred_point
    draw.ellipse((pred_x - 2, pred_y - 2, pred_x + 2, pred_y + 2), fill="red", outline="red")

    return image


transform = v2.Compose([v2.ToImage(),
                        v2.ToDtype(torch.float32, scale = True),
                        v2.Normalize(mean = (0.5,), std  = (0.5,))
])

path = r"C:\Users\admin\PycharmProjects\train_ii\train_pytorch\dataset"

data = sqDataset(path=path,transform=transform)

img, target = data[55]

x = img.reshape(-1,64*64)

model_state_dict = torch.load("model_square.pt")
model = model_square(64*64,2)
model.load_state_dict(model_state_dict)


model.eval()
pred = torch.round(model(x)).squeeze().detach().numpy()
target = target.numpy()
print(f"предсказание модели {pred}")
print(f"настоящая точка {target}")



img = F.to_pil_image(img)
img = draw_points(img, pred)
img.show()

