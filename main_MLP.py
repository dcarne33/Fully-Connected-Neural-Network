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
from mini_batch import split_batch
from NN_functions import initialize


if __name__ == "__main__":
    print("Welcome! This is a fully connected neural network to solve the Kaggle cardiovascular disease problem.")
    print("https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset?resource=download")
    # NN size, eg. a NN with 10 inputs, 2 hidden layers with 5 nodes each and 1 output node would be [10, 5, 5, 1]
    size = np.array([4, 300, 1])

    # activation function of each array, one for each non-input layer
    # options: "relu", "lrelu", "sig", "lin"
    activation = ["relu", "linear"]

    # epochs, number of iterations
    epochs = 100

    # Learning rate and momentum for backpropagation
    alpha = 0.3
    beta = 0.9

    # Number of batches
    batch = 5

    # name of input data
    name = str("CVD_cleaned.txt")

    # percent of data to train on as a decimal
    train_percent = 0.8

    # ADD IN CHECK FOR ERRORS

    # split input into training and validation and clean up categorical inputs
    # this function is specific to the dataset
    train, validate = import_file(name, train_percent)

    # split into mini batches
    train = split_batch(train, batch)

    # initialize NN
    layers = initialize(size, train, activation)




