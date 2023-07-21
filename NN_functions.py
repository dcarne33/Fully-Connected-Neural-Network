#  Author: Daniel Carne
#  Network setup for the MLP project.
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
from activation_functions import sigmoid, lrelu, relu, linear

# each layer is an instance
class Layer:
    def __init__(self, node_number_in, node_number, train_size, activation):
        # activation type
        self.activation = activation

        # weights and biases
        # weight array is (node in prev layer, current node)
        self.w = np.random.rand(node_number_in, node_number)-0.5
        self.b = np.random.rand(node_number) - 0.5

        # inputs (n) and output (z) of each node for each training point
        self.n = np.zeros((node_number, train_size))
        self.z = np.zeros((node_number, train_size))

        # gradients, store for each training point
        self.dcdn = np.zeros((node_number, train_size))
        self.dcdb = np.zeros((node_number, train_size))
        self.dcdw = np.zeros((node_number_in, node_number, train_size))

        # previous gradients for momentum
        self.dcdb_prev = np.zeros((node_number))
        self.dcdw_prev = np.zeros((node_number_in, node_number))


def forward_propagation(layers, input, size):
    # func map for activation
    activation_map = {"sig": sigmoid, "lrelu": lrelu, "relu": relu, "lin": linear}

    # input dimension (data points, inputs)
    # loop through each data point
    for point in range(len(input[:, 0])):
        # first layer is unique
        # loop through each node in the layer
        for i in range(size[1]):
            # clear out prev values
            layers[0].n[i, point] = 0
            # loop through each node in previous layer
            for j in range(size[0]):
                # n += z*w
                layers[0].n[i, point] += input[point, i] * layers[0].w[i, j]
            # add bias
            layers[0].n[i, point] += layers[0].b[i]
            # activation function
            layers[0].n[i, point] = activation_map[layers[0].activation](layers[0].n[i, point])


        # loop through each layer
        for layer in range(1, len(size)-1):
            # loop through each node in the layer
            for i in range(size[layer+1]):
                # clear out prev values
                layers[layer].n[i, point] = 0
                # loop through each node in previous layer
                for j in range(size[layer]):
                    # n += z*w
                    layers[layer].n[i, point] += layers[layer-1].z[i, point]*layers[layer].w[i, j]
                # add bias
                layers[layer].n[i, point] += layers[layer].b[i]
                # activation function
                layers[layer].n[i, point] = activation_map[layers[layer].activation](layers[layer].n[i, point])
    return layers


def backward_propagation(layers, input, size):
    # output layer first as this is unique
    # loop through each data point
    for point in range(len(input[:, 0])):
        # dcdn at single output node
        layers[len(size)-2].dcdn = 1

    return


def train_nn(layers, train, epochs, size):
    # number of batches
    batch_number = len(train[0, 0, :])

    # loop through each epoch
    for iter in range(epochs):
        # loop through each batch
        for batch in range(batch_number):
            # forward propagation
            layers = forward_propagation(layers, train[:, :, batch], size)

            # backward propagation


    return


def initialize(size, train, activation):
    # each object in layers is one class containing all information including weights and biases
    layers = []
    # initilize each layer
    for i in range(len(size)-1):
        layers.append(Layer(size[i], size[i+1], len(train[:, 0, 0]), activation[i]))


    return layers
