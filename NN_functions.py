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


def initialize(size, train):
    # max number of nodes in a layer
    max = 0
    for i in range(len(size)):
        if size[i] > max:
            max = size[i]


    ### MAKE THIS A CLASS WHERE EACH LAYER IS AN OBJECT WITH AN ARRAY OF APPROPRIATE SIZE


    # weights and biases array
    # first dimension is layer number, second dim. is incoming node, third dim. is outgoing node
    w = np.random.rand(len(size), max, max)-0.5
    # first dimension is layer number, second dim. is node
    b = np.random.rand(len(size), max)-0.5

    # inputs (n) and outputs (z) of each node
    # first dim. is layer number, second dim. is node number, third dim is for each training piece in a batch
    n = np.zeros(((len(size)), max, len(train[:, 0, 0])))
    z = np.zeros(((len(size)), max, len(train[:, 0, 0])))

    # store each gradient, same dim as prev
    dcdn = np.zeros((len(size), max, len(train[:, 0, 0])))
    dcdb = np.zeros((len(size), max, len(train[:, 0, 0])))
    dcdw = np.zeros((len(size), max, max, len(train[:, 0, 0])))
    # store previous gradients for momentum
    dcdb_prev = np.zeros((len(size), max))
    dcdw_prev = np.zeros((len(size), max, max))

    return
