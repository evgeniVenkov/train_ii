import numpy as np
import random
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()

dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]

loss_arr = []


input_dim = 4
output_dim = 3
h_dim = 10

x = np.random.rand(1, input_dim)
y = random.randint(0, output_dim - 1)

w1 = np.random.rand(input_dim, h_dim)
b1 = np.random.rand(1, h_dim)
w2 = np.random.rand(h_dim, output_dim)
b2 = np.random.rand(1, output_dim)




def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sparse_cross_e(z, y):
    return -np.log(z[0, y])


def to_full(y, num_class):
    y_full = np.zeros((1, num_class))
    y_full[0, y] = 1
    return y_full


def relu_der(t):
    return (t >= 0).astype(float)

def predict(x):
    t1 = x @ w1 + b1
    h1 = relu(t1)
    t2 = h1 @ w2 + b2
    z = softmax(t2)
    return z
def calc_acc():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct/len(dataset)
    return acc


ALPHA = 0.001
NUM_EPOCHS = 5



for ep in range(NUM_EPOCHS):
    for i in range(len(dataset)):
        #Forward
        t1 = x @ w1 + b1
        h1 = relu(t1)
        t2 = h1 @ w2 + b2
        z = softmax(t2)
        E = sparse_cross_e(z, y)

        #Backward
        y_full = to_full(y, output_dim)
        de_dt2 = z - y_full
        de_dw2 = h1.T @ de_dt2
        de_db2 = de_dt2
        de_dh1 = de_dt2 @ w2.T
        de_dt1 = de_dh1 * relu_der(t1)
        de_dw1 = x.T @ de_dt1
        de_db1 = de_dt1

        #update

        w1 = w1 - ALPHA * de_dw1
        b1 = b1 - ALPHA * de_db1
        w2 = w2 - ALPHA * de_dw2
        b2 = b2 - ALPHA * de_db2

        loss_arr.append(E)

acc = calc_acc()
print("Точность: " , acc)

plt.plot(loss_arr)
plt.show()

