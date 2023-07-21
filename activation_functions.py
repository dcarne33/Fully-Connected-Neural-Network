#  Author: Daniel Carne
#  Activation functions and their derivatives for the MLP project.
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


from numba import njit
import numpy as np


# Leaky Relu
@njit()
def lrelu(y, alpha):
    if y < 0:
        y = alpha*y
    return y

# Leaky Relu derivative
@njit()
def lrelu_der(y, alpha):
    if y <= 0:
        return alpha
    else:
        return 1

# Relu
@njit()
def relu(y, alpha):
    if y < 0:
        y = 0
    return y

# Relu derivative
@njit()
def relu_der(y, alpha):
    if y <= 0:
        return 0
    else:
        return 1


# sigmoid
@njit()
def sigmoid(y):
    return 1/(1+np.exp(-y))

# sigmoid derivative
@njit()
def sigmoid_der(y):
    der = sigmoid(y)*(1-sigmoid(y))
    return der

# linear
@njit()
def linear(y):
    return y

# linear derivative
@njit()
def linear_der(y):
    return 1