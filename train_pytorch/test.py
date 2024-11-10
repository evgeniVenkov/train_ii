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
