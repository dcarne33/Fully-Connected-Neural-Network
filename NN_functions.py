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

# each layer is an instance
class Layer:
    def __init__(self, node_number_in, node_number, train_size):
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


def forward_propagation(layers, train):
    return


def train_nn(layers, train, epochs):
    # number of batches
    batch_number = len(train[0, 0, :])
    return


def initialize(size, train):
    # each object in layers is one class containing all information including weights and biases
    layers = []
    # initilize each layer
    for i in range(len(size)-1):
        layers.append(Layer(size[i], size[i+1], len(train[:, 0, 0])))


    return layers
