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





# model = MyModel4(20,10)
# summary(model,(20,))

    print( round(1/3, 2) )
class BasicBlock(nn.Module):
    def __init__(self,inp,out):
        super().__init__()
        self.linear = nn.Linear(inp,10)
        self.act = nn.ReLU()
        self.linear1 = nn.Linear(10,out)
    def forward(self,x):
        x1 = x
        x = self.linear(x)
        x = self.act(x)
        x = self.linear1(x)
        x = self.act(x + x1)
        return x
class MyModel5(nn.Module):
    def __init__(self,inp,out):
        super().__init__()
        self.Basic = nn.Sequential(*[BasicBlock(inp,inp) for _ in range(10)])
        self.linear = nn.Linear(inp,out)
    def forward(self,x):
        x = self.Basic(x)
        return self.linear(x)
class MyModel6(nn.Module):
    def __init__(self,inp,out,out1):
        super().__init__()
        self.linear = nn.Linear(inp,10)
        self.linear1 = nn.Linear(10,out)
        self.linear2 = nn.Linear(out,10)
        self.linear3 = nn.Linear(10,out1)
        self.act = nn.ReLU()
    def forward(self,x):
        x = self.act(self.linear(x))

        x1 = self.linear1(x)

        x = self.act(x1)
        x = self.act(self.linear2(x))
        return x1, self.linear3(x)
class MyModel7(nn.Module):
    def __init__(self,inp,out1,out2,out3):
        super().__init__()
        self.out1 = nn.Sequential(* [nn.Linear(15,10),nn.ReLU(),nn.Linear(10,out1),nn.ReLU()])
        self.out2 = nn.Sequential(*[nn.Linear(15,10),nn.ReLU(),nn.Linear(10,out2),nn.ReLU()])
        self.lin_1 = nn.Sequential(*[nn.Linear(inp,10),nn.ReLU(),nn.Linear(10,15),nn.ReLU()])
        self.lin_2 = nn.Sequential(*[nn.Linear(15, 10), nn.ReLU(), nn.Linear(10, 15), nn.ReLU()])
        self.linear = nn.Linear(15,out3)

    def forward(self,x):
        x = self.lin_1(x)
        out1 = self.out1(x)
        x = self.lin_2(x)
        out2 = self.out2(x)
        out3 = self.linear(x)
        return [out1,out2,out3]
def get_block(inp,out,out1):
    return nn.Sequential(*[nn.Linear(inp,out), nn.ReLU(), nn.Linear(out,out1),nn.ReLU()])
class MyModel8(nn.Module):
    def __init__(self,inp,out):
        super().__init__()

        self.lin_1 = get_block(inp,10,15)
        self.lin_2 = get_block(15,10,7)
        self.linear = nn.Linear(7,7)
        self.act = nn.ReLU()
        self.lin_3 = get_block(7,10,15)
        self.lin_4 = get_block(15,10,20)
        self.linear1 = nn.Linear(20,out)
    def forward(self,x):
        x1 = self.lin_1(x)
        x2 = self.lin_2(x1)
        x = self.act(self.linear(x2))
        x = self.lin_3(x+x2)
        x = self.lin_4(x+x1)
        return self.linear1(x)

class MyModel9(nn.Module):
    def __init__(self, inp,out):
        super().__init__()
        self.forwar = nn.Sequential(
            nn.Linear(inp,52),
            nn.ReLU(),
            nn.Linear(52,26),
            nn.ReLU(),
            nn.Linear(26,out)
        )
    def forward(self, x):
        return self.forwar(x)

class MyModel01(nn.Module):
    def __init__(self, inp, inp1, out):
        super().__init__()

        self.linear = nn.Linear(inp, 10)
        self.act = nn.ReLU()
        self.linear1 = nn.Linear(10+inp1, 10)
        self.linear2 = nn.Linear(10, out)

    def forward(self, img, other):
        x = self.act(self.linear(img))

        cat = torch.cat((x, other), dim=1)
        out = self.act(self.linear1(cat))
        out = self.linear2(out)
        return out

# model = MyModel01(15*15,10,2)
#
# loss_model = nn.L1Loss()
# opt = torch.optim.Adam(model.parameters(), lr = 0.1)
# summary(model, input_size=(15*15,))
#
#
# # tensor = torch.rand(1, 15*15)
# # out = model(tensor)
# print(f'vvvvvvvvvvvvvvvvvvvvv\n{model} \n -------------------------')  # Вывод формы результата

print("CUDA доступен:" if torch.cuda.is_available() else "CUDA не доступен")

inp = 32
out = 64

h_i = 224
w_i = 224

pading = (0,0)
karnel_size = (1,1)
stride = (1,1)


H1 = ((h_i+2*pading[0] -1 *(karnel_size[0]-1)-1)/stride[0])+1
W1 = ((w_i+2*pading[1]-1*(karnel_size[1]-1)-1)/stride[1])+1

weights = (karnel_size[0] * karnel_size[1]* inp + 1)*out
print("---------------------")
print(H1,W1)
print(weights)

model = nn.Sequential()
model.add_module("1",nn.Conv2d(3,64,(3,3)))
model.add_module("1_1",nn.ReLU())
model.add_module("2",nn.Conv2d(64,64,(3,3)))


model = nn.Sequential(nn.Conv2d(1,5,(3,3)),
                      nn.ReLU(),
                      nn.Conv2d(5,10,(3,3)),
                      nn.ReLU(),
                      nn.Conv2d(10,15,(3,3))
                      )

print(model)