import torch

# Пример тензора
tensor = torch.randn(1, 20, 410, 645)

# Удаление первого слоя
tensor = tensor[:, 1:]

print(tensor.shape[1])
