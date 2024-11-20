import torch

l = [0,1,2,3,4,5,6,7,8,9]

tensor = torch.tensor(l)
tensor = torch.tensor(l,dtype = torch.float32)
tensor = torch.tensor(l,dtype = torch.float32 , requires_grad = True)
tensor_size = tensor.size()
tesnor_dtype = tensor.dtype

tensor_ones = torch.ones([2,4])
tensor_zeros = torch.zeros_like(tensor_ones)

new_tensor = tensor_ones[...,None]

tensor = torch.rand(64,28,28)

tensor = tensor[:,None,...]

tensor_1 = torch.rand(64,20)
tensor_2 = torch.rand(64,30)

tensor = torch.cat([tensor_1,tensor_2], dim = 1)


batch_tensor = torch.ones(10,28,28,1)

tensor = batch_tensor.view(batch_tensor.size(0),-1)

tensor = torch.tensor(([[17, 35],
        [16, 24],
        [36, 16],
        [17, 22],
        [31, 18]]))
max = torch.argmax(tensor, dim=1)

from torch.utils.data import Dataset
class myDataset(Dataset):
        def __int__(self):
                pass
        def __len__(self):
                pass
        def __getitem__(self, index):
                pass



print(max, end = "\n-----------------\n")
print(tensor_zeros)