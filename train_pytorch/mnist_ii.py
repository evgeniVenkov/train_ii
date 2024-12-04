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
import time

from datetime import datetime



device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
class MnistDataset(Dataset):
    def __init__(self, path, transform=None, cache_file="mnist_dataset_train.pt"):
        self.path = path
        self.transform = transform
        self.cache_file = cache_file

        if os.path.exists(self.cache_file):

            print(f"Загрузка обработанных данных из файла {self.cache_file}")
            data = torch.load(self.cache_file)
            self.data_list = data['data_list']
            self.targets = data['targets']
            self.classes = data['classes']
            self.class_to_idx = data['class_to_idx']
        else:

            print("Обработка данных...")
            self._process_and_cache_data()

    def _process_and_cache_data(self):
        self.data_list = []
        self.targets = []

        last_part = os.path.basename(self.path)
        TOTAL = 10000 if last_part == 'testing' else 60000


        path_loop = tqdm(total=TOTAL, desc="Обработка изображений")
        for path_dir, dir_list, file_list in os.walk(self.path):
            if path_dir == self.path:
                self.classes = dir_list
                self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
                continue

            cls = path_dir.split(os.sep)[-1]
            for name_file in file_list:
                file_path = os.path.join(path_dir, name_file)
                sample = np.array(Image.open(file_path).resize((28, 28)), dtype=np.float32)
                self.data_list.append(sample)
                self.targets.append(self.class_to_idx[cls])
                path_loop.update()

        path_loop.close()


        self.data_list = torch.tensor(np.array(self.data_list), dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.long)


        data = {
            'data_list': self.data_list,
            'targets': self.targets,
            'classes': self.classes,
            'class_to_idx': self.class_to_idx
        }
        torch.save(data, self.cache_file)
        print(f"Обработанные данные сохранены в файл: {self.cache_file}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target
class my_model(nn.Module):
    def __init__(self,input,output):
        super().__init__()
        self.layer1 = nn.Linear(input,128)
        self.layer2 = nn.Linear(128,output)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.layer1(x)
        x = self.act(x)
        out = self.layer2(x)

        return out
def get_data(Batch_size = 1):
    path_test = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'mnist\testing')
    path_train = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'mnist\training')


    train_dataset = MnistDataset(path_train)
    test_dataset = MnistDataset(path_test,None,"mnist_dataset_test.pt")

    train_data, val_data = random_split(train_dataset, [0.8, 0.2])

    train_loader = DataLoader(train_data, batch_size=Batch_size,shuffle=True)
    val_loader = DataLoader(val_data, batch_size=Batch_size, shuffle=False)
    test_Loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False)

    return train_loader, val_loader, test_Loader
def log_():
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    log = (f"\nMEAN COUNT {mean_count}\n"
           f"Train_loss: {np.mean(run_train_loss[mean_count:]):.4f}, "
           f"Train accuracy: {np.mean(accuracy_train[mean_count:]):.4f}\n"
           f"val_loss: {np.mean(run_val_loss[mean_count:]):.4f}, Val accuracy: {np.mean(accuracy_val[mean_count:]):.4f}\n"
           f"learning rate start: {LearningRate}:  |  end: {opt.param_groups[0]['lr']:.6f},\n"
           f"batch_size: {Batch_size}\n"
           f"Elapsed time {end_time - start_time}\n"
           f"Epochs: {EPOHS} |     "
           f"time {time} \n"
           f"-----------------------------------------------------------------------------------------------")

    with open("log", "a") as f:
        f.write(log)

Batch_size = 128
LearningRate = 0.001
EPOHS = 1

loss_fn = nn.CrossEntropyLoss()
model = my_model(784,10).to(device)
opt = torch.optim.Adam(model.parameters(), lr = LearningRate)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.01, patience=50, mode="min")

train_data,val_data,test_data = get_data(Batch_size)




run_train_loss = []
run_val_loss = []
accuracy_train =[]
accuracy_val =[]

start_time = time.time()

for i in range(EPOHS):

    model.train()

    train_loop = tqdm(train_data, leave = True)
    correct = 0
    total = 0
    for x, target in train_loop:



        x = x.reshape(-1,28*28).to(device)
        target = target.reshape(-1)
        target = torch.eye(10)[target].to(device).to(torch.float32)


        pred = model(x)

        correct += (pred.argmax(dim =1) == target.argmax(dim = 1)).sum().item()
        total += target.size(0)


        loss = loss_fn(pred,target)
        opt.zero_grad()
        loss.backward()


        opt.step()
        if i > 3:
            lr_scheduler.step(loss.item())


        run_train_loss.append(loss.item())
        mean_train_loss = sum(run_train_loss)/ len(run_train_loss)

        accuracy_train.append(correct/total)
        train_loop.set_description(f"Eposh {i+1} loss: {mean_train_loss:.4f}, "
                                   f"accuracy: {correct/total:.4f}, "
                                   f"learningrate{opt.param_groups[0]['lr']}")


    model.eval()


    correct =0
    total =0
    val_loop = tqdm(val_data,leave =False)
    with torch.no_grad():
        for x, target in val_loop:
            x = x.reshape(-1,28*28).to(device)

            target = target.reshape(-1).to(torch.int64).to(device)

            pred = model(x)

            predict = torch.argmax(pred,dim=1)
            correct+= (predict == target).sum().item()
            total += target.size(0)
            accuracy_val.append(correct/total)

            loss = loss_fn(pred,target)



            run_val_loss.append(loss.item())
            mean_val_loss = sum(run_val_loss)/len(run_val_loss)
            val_loop.set_description(f"val loss: {mean_val_loss:.4f},"
                                     f"accuracy:{correct/total:.4f},"
                                     f"learningrate{opt.param_groups[0]['lr']}")


end_time = time.time()
mean_count = -100

log_()

print(f"Elapsed time {end_time - start_time}")

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

