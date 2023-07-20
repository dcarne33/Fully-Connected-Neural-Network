#  Author: Daniel Carne
#  Fully connected neural network (MLP). Currently only setup for one output node.
#  Copyright (C) 2023 Daniel Carne <dandaman35@gmail.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np
from numba import njit, prange
from matplotlib import pyplot as plt
import csv
import math

# NN size, eg. a NN with 10 inputs, 2 hidden layers with 5 nodes each and 1 output node would be [10, 5, 5, 1]
size = [4, 300, 1]
size = np.asarray(size)

# epochs, number of iterations
epochs = 100000
erplot = np.zeros((epochs, 2))

# finds maximum sizes to make arrays
max = 0
for i in range(len(size)):
    if size[i] > max:
        max = size[i]

# Learning rate and momentum
alpha = 0.3
beta = 0.9

# Number of batches. Number of training points must be divisible by this number.
batch = 1

# imports inputs and answers from CSVs. Must be stored in the same folder as this python file.
# All data must be normalized to between 0-1 if using a sigmoidal function.
# If using Leaky ReLu, normalization is not as important. Read on Batch Normalization (BN) if there are issues.
a = []
with open("input.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        a.append(row)
a = np.asarray(a)

input = a[:3000]
inputVAL = a[3000:4000]

b = []
with open("answer.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        b.append(row)
b = np.asarray(b)
answer = b[:3000]
answerVAL = b[3000:4000]
# Number of training data points
trainsize = int(len(answer))
trainsizeVAL = int(len(answerVAL))

error = np.zeros(int(len(answer)))
errorVAL = np.zeros(int(len(answerVAL)))

# arrays containing weights and biases
w = np.random.rand(max, max, int(len(size)-1))-0.5
b = np.random.rand(max, int(len(size)))-0.5

# inputs (n) and outputs (z) of each node
n = np.zeros((max, int(len(size)), trainsize))
z = np.zeros((max, int(len(size)), trainsize))
zVAL = np.zeros((max, int(len(size)), trainsizeVAL))
for t in range(trainsize):
    for j in range(size[0]):
        z[j, 0, t] = input[t, j]

for t in range(trainsizeVAL):
    for j in range(size[0]):
        zVAL[j, 0, t] = inputVAL[t, j]


# gradients of each node. Storing these for each training point separately so I can run them in parallel.
dcdn = np.zeros((max, int(len(size)), trainsize))
dcdw = np.zeros((max, max, int(len(size)-1), trainsize))
dcdb = np.zeros((max, int(len(size)), trainsize))
# previous gradients stored for momentum
dcdwp = np.zeros((max, max, int(len(size)-1)))
dcdbp = np.zeros((max, int(len(size))))

# Leaky Relu
@njit()
def activate(y):
    if y < 0:
        y = 0.3*y
    return y


@njit()
def activate_der(y):
    if y <= 0:
        y = 0.3
    else:
        y = 1
    return y


# Sigmoidal
#@njit()
#def activate(x):
#    x = 1/(1+math.exp(-x))
#    return x


# sigmoid derivative
#@njit()
#def activate_der(x):
#    x = 1/(1+math.exp(-x))
#    return x*(1-x)


# forward propagation function
@njit(parallel=True)
def forward(w, b, n, z, trainsize, size, error, answer, i2, batch):
    # clears out previous summation
    for t in prange(int(i2*trainsize/batch), int(trainsize/batch*(i2+1))):
        for i in range(1, int(len(size))):
            for j in range(size[i]):
                n[j, i, t] = 0

    # forward propagation
    for t in prange(int(i2*trainsize/batch), int(trainsize/batch*(i2+1))):
        for i in range(1, int(len(size))):
            for j in range(size[i]):
                for k in range(size[i-1]):
                    n[j, i, t] += z[k, i-1, t]*w[k, j, i-1]
                n[j, i, t] += b[j, i]
                z[j, i, t] = activate(n[j, i, t])

    for t in prange(int(i2*trainsize/batch), int(trainsize/batch*(i2+1))):
        error[t] = ((answer[t, 0] - z[0, int(len(size)-1), t])**2)**0.5

    return z, n, error


# backward propagation function
@njit(parallel=True)
def backward(dcdn, dcdw, dcdb, z, n, b, w, answer, trainsize, size, beta, alpha, dcdbp, dcdwp, i2, batch):
    # clears gradients at each node
    for t in prange(int(i2*trainsize/batch), int(trainsize/batch*(i2+1))):
        for i in range(1, int(len(size))):
            for j in range(size[i]):
                dcdn[j, i, t] = 0
    # This is for the output layer nodes which are unique.
    for t in prange(int(i2*trainsize/batch), int(trainsize/batch*(i2+1))):
            for j in range(size[int(len(size)-1)]):
                # calculates gradient at each node in the output layer
                dcdn[j, int(len(size)-1), t] = (z[j, int(len(size)-1), t] - answer[t, 0]) * activate_der(z[j, int(len(size)-1), t])
                #  calculates gradient for each bias in the output layer
                dcdb[j, int(len(size)-1), t] = dcdn[j, int(len(size)-1), t]
                for k in range(size[int(len(size)-2)]):
                    # calculates gradient of each weight going into the output layer
                    dcdw[k, j, int(len(size)-2), t] = dcdn[j, int(len(size)-1), t] * z[k, int(len(size)-2), t]

    # This is for the rest of the nodes, which are not unique.
    for t in prange(int(i2*trainsize/batch), int(trainsize/batch*(i2+1))):
        for i in range(int(len(size)-2), 0, -1):
            for j in range(size[i]):
                for k in range(size[i+1]):
                    dcdn[j, i, t] += dcdn[k, i+1, t] * w[j, k, i] * activate_der(z[j, i, t])
                dcdb[j, i, t] = dcdn[j, i, t]
                for k in range(size[i - 1]):
                    dcdw[k, j, i-1, t] = dcdn[j, i, t] * z[k, i-1, t]

    # Now that we know the gradients we can apply them to change w and b accordingly.
    # First sum up the gradients from each training data point.
    for t in range(int(i2*trainsize/batch), int(trainsize/batch*(i2+1))):
        for i in range(int(len(size)-1), 0, -1):
            for j in prange(size[i]):
                dcdb[j, i, 0] += dcdb[j, i, t]
                for k in range(size[i-1]):
                    dcdw[k, j, i-1, 0] += dcdw[k, j, i-1, t]

    for i in range(int(len(size)-1), 0, -1):
        for j in prange(size[i]):
            b[j, i] -= alpha * (dcdb[j, i, 0] / (trainsize/batch)) + beta * dcdbp[j, i]
            dcdbp[j, i] = alpha * (dcdb[j, i, 0] / (trainsize/batch)) + beta * dcdbp[j, i]
            for k in range(size[i-1]):
                w[k, j, i - 1] -= alpha * (dcdw[k, j, i - 1, 0] / (trainsize/batch)) + beta * dcdwp[k, j, i-1]
                dcdwp[k, j, i-1] = alpha * (dcdw[k, j, i - 1, 0] / (trainsize/batch)) + beta * dcdwp[k, j, i-1]

    return w, b, dcdwp, dcdbp


def main(dcdn, dcdw, dcdb, z, n, b, w, answer, trainsize, size, beta, alpha, error, erplot, dcdbp, dcdwp, batch, answerVAL, inputVAL, trainsizeVAL, errorVAL, zVAL):
    alpha = 0.001
    for i in range(epochs):
        for i2 in range(batch):
            if i == 100:
                alpha = 0.05

            if i == 30000:
                alpha = 0.005
            z, n, error = forward(w, b, n, z, trainsize, size, error, answer, i2, batch)

            w, b, dcdwp, dcdbp = backward(dcdn, dcdw, dcdb, z, n, b, w, answer, trainsize, size, beta, alpha, dcdbp, dcdwp, i2, batch)

            # validation
            zVAL, n, errorVAL = forward(w, b, n, zVAL, trainsizeVAL, size, errorVAL, answerVAL, i2, batch)

        erplot[i, 0] = np.average(error)
        erplot[i, 1] = np.average(errorVAL)
        print('Epoch:', i, 'Error:', np.average(error))
    return z, erplot, w, b


z, erplot, w, b = main(dcdn, dcdw, dcdb, z, n, b, w, answer, trainsize, size, beta, alpha, error, erplot, dcdbp, dcdwp, batch, answerVAL, inputVAL, trainsizeVAL, errorVAL, zVAL)

np.save('weights', w)
np.save('biases', b)
plt.plot(erplot[10:, 0], label="Train")
plt.plot(erplot[10:, 1], label="Validate")
plt.yscale("log")
plt.legend()
plt.show()
for i in range(2000):
    if np.abs(z[0, int(len(size)-1), i] - answer[i]) > 0.03:
            print(z[0, 0, i], z[1, 0, i], z[2, 0, i], z[3, 0, i])
