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

inp = 20
out = 50

h_i = 197
w_i = 297

pading = (0,0)
karnel_size = (1,1)
stride = (2,2)


H1 = ((h_i+2*pading[0] -1 *(karnel_size[0]-1)-1)/stride[0])+1
W1 = ((w_i+2*pading[1]-1*(karnel_size[1]-1)-1)/stride[1])+1

weights = (karnel_size[0] * karnel_size[1]* inp + 1)*out
print("---------------------")
print(H1,W1)
print(weights)

exit()


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

class myConvModel(nn.Module):
    def __init__(self,inp,out):
        super().__init__()
        self.linear = nn.Conv2d(inp,5,(3,3))
        self.linear1 = nn.Conv2d(5,10,(3,3))
        self.linear2 = nn.Conv2d(10,out,(3,3))
        self.act = nn.ReLU()
    def forward(self,x):
        x = self.linear(x)
        x = self.act(x)
        x = self.act(self.linear1(x))
        return self.linear2(x)
class myConvModel2(nn.Module):
    def __init__(self,inp,):
        super().__init__()
        self.linear_left = nn.Conv2d(inp,10,(1,1))
        self.linear_right = nn.Conv2d(inp,10,(3,3),padding=(1,1))
        self.act = nn.ReLU()
    def forward(self,x):
        x1 = self.act(self.linear_left(x))
        x2 = self.act(self.linear_right(x))
        return torch.cat([x1,x2], dim = 1),x1,x2
class myConvModel3(nn.Module):
    def __init__(self,inp,out):
        super().__init__()
        self.linear = nn.Conv2d(inp,out,(3,3))
        self.act = nn.ReLU()
    def forward(self,x):
        x = torch.cat(x,dim = 1)
        return  self.act(self.linear(x))
class myConvModel4(nn.Module):
    def __init__(self,inp,out):
        super().__init__()
        self.Conv = nn.Conv2d(inp,out,(3,3),(1,1),(1,1),bias=False)
        self.act  = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(out)

    def forward(self,x):
        x = self.Conv(x)
        x = self.act(self.batchnorm(x))
        return x
class myConvModel5(nn.Module):
    def __init__(self,inp,out):
        super().__init__()
        self.Conv = nn.Conv2d(inp,10,(1,1),bias=False)
        self.act = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(10)
        self.Conv1 = nn.Conv2d(10,out,(3,3),(1,1),(1,1),bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out)
    def forward(self,x):
        x = self.batchnorm(self.Conv(x))
        x = self.Conv1(self.act(x))
        x = self.act(self.batchnorm1(x))
        return x
class myConvModel6(nn.Module):
    def __init__(self,inp,out):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, 10,(3,3),padding = (1,1),bias = False)
        self.batchnorm = nn.BatchNorm2d(10)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(10,out,(3,3),padding = (1,1),bias = False)
    def forward(self,x):
        x1 = self.batchnorm(self.conv1(x))
        x1 = self.conv2(self.act(x1))
        x1= self.batchnorm(x1)
        x = self.act(x + x1)
        return x
class myConvModel7(nn.Module):
    def __init__(self,inp,out):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(10,10,(1,1),(2,2),bias = False),
                                  nn.BatchNorm2d(10))
        self.main = nn.Sequential(nn.Conv2d(inp,10,(3,3),(2,2),(1,1),bias = False),
                                  nn.BatchNorm2d(10),
                                  nn.ReLU(),
                                  nn.Conv2d(10,out,(3,3),padding = (1,1), bias = False),
                                  nn.BatchNorm2d(out))
        self.act = nn.ReLU()
    def forward(self,x):
        x1 = self.main(x)
        x2 = self.down(x)
        return self.act(x1 + x2)
class Bottleneck(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(inp, 7, (1, 1), bias=False),
                                  nn.BatchNorm2d(7),
                                  nn.ReLU(),
                                  nn.Conv2d(7, 7, (3, 3), padding=(1, 1), bias=False),
                                  nn.BatchNorm2d(7),
                                  nn.ReLU(),
                                  nn.Conv2d(7, out, (1, 1), bias=False),
                                  nn.BatchNorm2d(out))
        self.act = nn.ReLU()

    def forward(self, x):
        x1 = self.main(x)
        return self.act(x + x1)
class MyModel(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.one = nn.Sequential(nn.Conv2d(inp, 5, (3, 3)),
                                 nn.ReLU(),
                                 nn.Conv2d(5, 5, (3, 3), padding=(1, 1)),
                                 nn.ReLU())
        self.linear = nn.Linear(out, 10)

    def forward(self, x):
        x = self.one(x)
        x = x.flatten(start_dim=1)
        return self.linear(x)
class BasicBlock(nn.Module):
    def __init__(self,inp,out):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(1,1,(1,1),(2,2),bias = False),
                                  nn.BatchNorm2d(1))
        self.main = nn.Sequential(nn.Conv2d(inp,7,(1,1),bias = False),
                                  nn.BatchNorm2d(7),
                                  nn.ReLU(),
                                  nn.Conv2d(7, 7, (3, 3),(2,2),(1,1), bias=False),
                                  nn.BatchNorm2d(7),
                                  nn.ReLU(),
                                  nn.Conv2d(7,out,(1,1), bias = False),
                                  nn.BatchNorm2d(out))
        self.act = nn.ReLU()
    def forward(self,x):
        x1 = self.main(x)
        x2 = self.down(x)
        return self.act(x1 + x2)
class pool(nn.Module):
    def __init__(self,inp):
        super().__init__()
        self.run = nn.Sequential(nn.Conv2d(inp,5,(7,7),(2,2),(3,3),bias = False),
                                 nn.BatchNorm2d(5),
                                 nn.ReLU(),
                                 nn.MaxPool2d((3,3),(2,2),(1,1)))
    def forward(self,x):
        return self.run(x)
class Transition(nn.Module):
    def __init__(self,inp,out):
        super().__init__()
        self.run = nn.Sequential(nn.BatchNorm2d(inp),
                                 nn.ReLU(),
                                 nn.Conv2d(inp,out,(1,1),bias = False))
                                 # nn.AvgPool2d((2,2),2))
    def forward(self,x):
        return self.run(x)
class ModelAverage(nn.Module):
    def __init__(self,inp,out):
        super().__init__()
        self.run = nn.Sequential(nn.Conv2d(inp,out,(3,3),padding=(1,1),bias = False),
                                 nn.BatchNorm2d(out),
                                 nn.ReLU(),
                                 nn.AvgPool2d((10,10)))
    def forward(self,x):
        return self.run(x)
class FinalModel(nn.Module):
    def __init__(self,inp,out):
        super().__init__()
        self.one = nn.Sequential(nn.Conv2d(inp,10,(3,3),bias = False),
                                 nn.BatchNorm2d(10),
                                 nn.ReLU(),
                                 nn.AvgPool2d((8,8)))
        self.linear = nn.Linear(10,out)

    def forward(self,x):
        x = self.one(x)
        x = x.flatten(start_dim=1)
        return self.linear(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
mas = []


model = FinalModel(5,5).to(device)
tensor = torch.rand(1,5,10,10).to(device)

x = model(tensor)
print(x.shape)
exit()


x = model(tensor)
summary(model, input_size=(5, 150, 150))

print(x.shape)
