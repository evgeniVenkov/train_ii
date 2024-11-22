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

import torch.nn as nn
from torchsummary import summary

model = nn.Sequential(nn.Linear(128,64),
                      nn.ReLU(),
                      nn.Linear(64,32))

model = nn.Sequential(nn.Linear(1024,512),
                      nn.ReLU(),
                      nn.Linear(512,256),
                      nn.ReLU(),
                      nn.Linear(256,128)
                      )
class myModel(nn.Module):
    def __init__(self,inp,out):
        super().__init__()
        self.Linear = nn.Linear(inp,52)
        self.act = nn.ReLU()
        self.Linear2 = nn.Linear(52,26)
        self.out = nn.Linear(26,out)

    def forward(self,x):
        x = self.Linear(x)
        x = self.act(x)
        x = self.Linear2(x)
        x = self.act(x)
        x = self.out(x)
        return x

class myModel1(nn.Module):
    def __init__(self,inp,inp1,out):
        super().__init__()
        self.inp1 = inp1
        self.Linear = nn.Linear(inp,52)
        self.act = nn.ReLU()
        self.Linear2 = nn.Linear(52,26)
        self.out = nn.Linear(26,out)

    def forward(self,x):
        x = self.Linear(x)
        x = self.act(x)
        x = self.Linear2(x+self.inp1)
        x = self.act(x)
        x = self.out(x)
        return x

class myModel2(nn.Module):
    def __init__(self,inp1,inp2,out):
        super().__init__()
        self.inp = inp2
        self.linear = nn.Linear(inp1,72 - inp2)
        self.act = nn.ReLU()
        self.linear1 = nn.Linear(72,26)
        self.out = nn.Linear(26, out)

    def forward(self,x,y):
        x= self.linear(x)
        x = self.act(x)
        x = torch.cat([x,y],dim = 1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.out(x)
        return x

class MyModel3(nn.Module):
    def __init__(self,inp1,inp2,out):
        super().__init__()
        self.linear = nn.Linear(inp1,52)
        self.linear1 = nn.Linear(52,26)
        self.linear2 = nn.Linear(52,out)
        self.act = nn.ReLU()

    def forward(self,x,y):
        x = self.linear(x)
        y = self.linear(y)
        x = self.act(x)
        y = self.act(y)
        x = self.linear1(x)
        y = self.linear(y)
        conc = torch.cat([x,y],dim = 1)
        out = self.linear2(conc)
        return out

class MyModel4(nn.Module):
    def __init__(self,inp,out):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(10):
            if i == 9:
                self.layers.add_module(f'layer_{i}', nn.Linear(inp - i, out))
                continue
            self.layers.add_module(f'layer_{i}', nn.Linear(inp-i,inp-i-1))
            self.layers.add_module(f"act_{i}",nn.ReLU())

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x


for i in range(10,20):
    print(i)

exit()


model = MyModel4(20,10)
summary(model,(20,))

print(model)