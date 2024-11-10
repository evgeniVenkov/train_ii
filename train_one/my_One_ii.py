import numpy as np

input_dim = 4
h_dim = 7
output_dim = 3

def relu(t):
    return  np.maximum(0,t)

def softmax(h):
    print(f"вход {h}")
    print(f"максимум {np.max(h)}")

    exp = np.exp(h)
    print(f"экспонента {exp}")
    print(f"экс минус макс {np.exp(h-np.max(h))}")

    return exp/np.sum(exp)


x = np.random.rand(input_dim)

w1 = np.random.rand(input_dim, h_dim)
b1 = np.random.rand(h_dim)

w2 = np.random.rand(h_dim, output_dim)
b2 = np.random.rand(output_dim)

t1 = x @ w1 + b1
h1 = relu(t1)
t2 = h1 @ w2 + b2
z = softmax(t2)

print(z)

