

import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm


class SequentialDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.data_list = []
        self.targets = []

        TOTAL = 118  # Количество изображений для отображения процесса загрузки
        path_loop = tqdm(total=TOTAL, desc='Loading data')

        for path_dir, dir_list, file_list in os.walk(self.path):
            if path_dir == self.path:
                self.classes = dir_list  # Классы (папки с изображениями)
                print(dir_list)
                self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
                continue

            cls = path_dir.split(os.sep)[-1]
            for name_file in sorted(file_list):  # Сортируем файлы для последовательности
                file_path = os.path.join(path_dir, name_file)
                image = Image.open(file_path).resize((645, 410))
                if self.transform:
                    image = self.transform(image)
                else:
                    image = transforms.ToTensor()(image)
                self.data_list.append(image)
                self.targets.append(self.class_to_idx[cls])
                path_loop.update()

        path_loop.close()

        # Перемещаем первые 48 фотографий в конец списка
        self.data_list = torch.stack(self.data_list)  # Преобразуем в один тензор
        self.targets = torch.tensor(self.targets, dtype=torch.long)

        self.data_list = torch.cat((self.data_list[48:], self.data_list[:48]))  # Переносим первые 48 в конец
        self.targets = torch.cat((self.targets[48:], self.targets[:48]))  # Переносим метки

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        target = self.targets[idx]
        return sample, target
class val_dataset(Dataset):
    def __init__(self,):
        self.data = []
        self.targets = []


    def add_data(self, data, target):
        self.data.append(data)
        self.targets.append(target)
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx], self.targets[idx]

path = r"C:\Users\admin\PycharmProjects\train_ii\pet\load_runner\data"


data = SequentialDataset(path)

img, tar = data[113]

print(tar)