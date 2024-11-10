import numpy as np
import matplotlib.pyplot as plt
import random

dataset = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]]

input_dim = 2
h_dim = 10
output_dim = 1
alpha = 0.1
epohs = 200

w1 = np.random.randn(input_dim, h_dim)
w2 = np.random.randn(h_dim, output_dim)
b1 = np.random.randn(h_dim)
b2 = np.random.randn(output_dim)

loss_arr = []
def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu_der(t):
    return (t >= 0).astype(float)
def sparse(z,y):
    return -np.log(z) if y == 1 else -np.log(1 - z)
def prognoz(x,y):
    t1 = x @ w1 + b1
    h1 = relu(t1)
    t2 = h1 @ w2 + b2
    z = sigmoid(t2)
    e = sparse(z, y)
    return z,e


for i in range(epohs):
    loss_epoh = 0
    random.shuffle(dataset)
    for data in dataset:
        x = np.array(data[:2]).reshape(1, -1)
        y = data[2]
    #forward
        t1 = x @ w1 + b1
        h1 = relu(t1)
        t2 = h1 @ w2 + b2
        z = sigmoid(t2)
        e = sparse(z,y)
        loss_epoh += e
    #back


        # dt2: ошибка выходного слоя
        dt2 = z - y  # dt2 будет скаляром

        dw2 = h1.T @ dt2
        db2 = dt2
        dh1 = dt2 @ w2.T

        dt1 = dh1 * relu_der(t1)

        dw1 = x.T @ dt1
        db1 = dt1

        #update

        w1 -= alpha * dw1
        w2 -= alpha * dw2
        b1 = b1 - alpha * db1
        b2 = b2 -  alpha * db2

    loss_avg = loss_epoh / len(dataset)
    loss_arr.append(loss_avg.item())

print(f'Finish loss {loss_arr[-1]}')
for data in dataset:
    x = np.array(data[:2]).reshape(1, -1)
    y = data[2]
    z, e = prognoz(x,y)
    print(f"Input: {data[:2]}, True Output: {y}, Predicted Output: {z}")

plt.plot(loss_arr)
plt.show()