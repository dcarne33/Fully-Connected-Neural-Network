#  Author: Daniel Carne
#  Fully connected neural network (MLP) for single output node problems.
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
from input_file import import_file

if __name__ == "__main__":
    print("Welcome! This is a fully connected neural network to solve the K.")

    # NN size, eg. a NN with 10 inputs, 2 hidden layers with 5 nodes each and 1 output node would be [10, 5, 5, 1]
    size = np.array([4, 300, 1])

    # epochs, number of iterations
    epochs = 100

    # Learning rate and momentum for backpropagation
    alpha = 0.3
    beta = 0.9

    # Number of batches
    batch = 1

    # name of input data
    name = str("CVD_cleaned.txt")

    # percent of data to train on as a decimal
    train_percent = 0.8

    # split input into training and validation and clean up categorical inputs
    train, validate = import_file(name, train_percent)

