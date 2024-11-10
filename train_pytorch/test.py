import torchvision
import os
import numpy as np
from PIL import Image
import json

def write_dataset(dataset, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Создание классов
    classes = {i: f"class_{i}" for i in range(10)}
    output_dirs = [os.path.join(output_dir, classes[i]) for i in range(10)]

    for dir in output_dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    for i, (image, label) in enumerate(dataset):
        output_filename = os.path.join(output_dirs[label], f"{i}.jpg")
        print("Запись: " + output_filename)

        # Сохранение изображения
        image.save(output_filename)


output_path = "./mnist"

# Загрузка MNIST
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True)

# Запись данных
write_dataset(train_data, os.path.join(output_path, "training"))
write_dataset(test_data, os.path.join(output_path, "testing"))


if not os.path.isdir("dataset"):
    os.mkdir("dataset")

img = np.random.randint(0, 50, [100000,64,64], dtype=np.uint8)
square = np.random.randint(100,200,[100000,15,15], dtype=np.uint8)

coords = np.empty([100000,2])
data= {}

for i in range(img.shape[0]):
    x = np.random.randint(20,44)
    y = np.random.randint(20,44)

    img[i,(y-7):(y+8),(x-7):(x+8)] = square[i]
    coords[i] = [x,y]

    name_img = f'img_{i}.jpeg'
    path_img = os.path.join('dataset/',name_img)

    image = Image.fromarray(img[i])
    image.save(path_img)

    data[name_img] = [x,y]
    print(f"{name_img} saved")

with open('dataset/coords.json','w') as f:
    json.dump(data,f,indent=2)
