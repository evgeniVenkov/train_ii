import numpy as np
import matplotlib.pyplot as plt
import random

def sigmoid(x):
    return 1/(1+np.exp(-x))

def cross_e(y, z):
    return -np.log(z) if y == 1 else -np.log(1 - z)

def sig_der(t):
    sig = sigmoid(t)
    return sig * (1 - sig)


dataset = [[1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]]

w = np.random.uniform(-0.1, 0.1, (2, 1))
b = np.random.uniform(-0.1, 0.1, (1, 1))
loss_arr = []
epohs = 1000
alfa = 0.1


for i in range(epohs):
    random.shuffle(dataset)
    loss_epoh = 0
    for data in dataset:
        x = np.array(data[:2]).reshape(1, -1)
        y = data[2]

#forward
        t = x @ w + b
        z = sigmoid(t)
        e = cross_e(y,z)
        loss_epoh += e

#backward
        dz = z - y
        dt = dz * sig_der(t)
        dw = x.T @ dt
        db = dt
#update
        w -= alfa * dw
        b -= alfa * db
    avg_loss = loss_epoh / len(dataset)
    loss_arr.append(avg_loss.item())


plt.plot(loss_arr)
plt.show()

print(f"final result x: {x}\ny: {y} \nz: {z}")

for data in dataset:
    x = np.array(data[:2]).reshape(1, -1)
    y = data[2]
    z = sigmoid(x @ w + b)
    print(f"Input: {data[:2]}, True Output: {y}, Predicted Output: {z}")
