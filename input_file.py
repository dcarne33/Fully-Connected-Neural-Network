#  Author: Daniel Carne
#  Input file and cleaning for the MLP project.
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
import csv


def import_file(name, train_percent):
    input = np.loadtxt(name, delimiter="\t", dtype=str, skiprows=1)
    # second array that contains non-string post-processed values
    data = np.zeros((len(input[:, 0]), len(input[0, :])))

    # categorize inputs, this is specific to the cardiovascular disease file
    for i in range(len(input[:, 0])):
        # categorize general health
        if input[i, 0] == "Poor":
            data[i, 0] = 0
        elif input[i, 0] == "Fair":
            data[i, 0] = 1
        elif input[i, 0] == "Good":
            data[i, 0] = 2
        elif input[i, 0] == "Very Good":
            data[i, 0] = 3
        elif input[i, 0] == "Excellent":
            data[i, 0] = 4

        # categorize last checkup
        if input[i, 1] == "Within the past year":
            data[i, 1] = 0
        elif input[i, 1] == "Within the past 2 years":
            data[i, 1] = 1
        elif input[i, 1] == "5 or more years ago":
            data[i, 1] = 5
        elif input[i, 1] == "Within the past 5 years":
            data[i, 1] = 4

        # categorize exercise
        if input[i, 2] == "No":
            data[i, 2] = 0
        elif input[i, 2] == "Yes":
            data[i, 2] = 1

        # categorize heart disease
        if input[i, 3] == "No":
            data[i, 3] = 0
        elif input[i, 3] == "Yes":
            data[i, 3] = 1

        # categorize skin cancer
        if input[i, 4] == "No":
            data[i, 4] = 0
        elif input[i, 4] == "Yes":
            data[i, 4] = 1

        # categorize other cancers
        if input[i, 5] == "No":
            data[i, 5] = 0
        elif input[i, 5] == "Yes":
            data[i, 5] = 1

        # categorize depression
        if input[i, 6] == "No":
            data[i, 6] = 0
        elif input[i, 6] == "Yes":
            data[i, 6] = 1

        # categorize diabetes
        if input[i, 7] == "No":
            data[i, 7] = 0
        elif input[i, 7] == "Yes":
            data[i, 7] = 2
        else:
            data[i, 7] = 1

        # categorize arthritis
        if input[i, 8] == "No":
            data[i, 8] = 0
        elif input[i, 8] == "Yes":
            data[i, 8] = 1

        # categorize sex
        if input[i, 9] == "Female":
            data[i, 9] = 0
        elif input[i, 9] == "Male":
            data[i, 9] = 1

        # age category
        if input[i, 10] == "18-24":
            data[i, 10] = 21
        elif input[i, 10] == "25-29":
            data[i, 10] = 27
        elif input[i, 10] == "30-34":
            data[i, 10] = 32
        elif input[i, 10] == "35-39":
            data[i, 10] = 37
        elif input[i, 10] == "40-44":
            data[i, 10] = 42
        elif input[i, 10] == "45-49":
            data[i, 10] = 47
        elif input[i, 10] == "50-54":
            data[i, 10] = 52
        elif input[i, 10] == "55-59":
            data[i, 10] = 57
        elif input[i, 10] == "60-64":
            data[i, 10] = 62
        elif input[i, 10] == "65-69":
            data[i, 10] = 67
        elif input[i, 10] == "70-74":
            data[i, 10] = 72
        elif input[i, 10] == "75-79":
            data[i, 10] = 77
        elif input[i, 10] == "80+":
            data[i, 10] = 80



    # split data up into train and validate sets
    train_data = np.round(len(data[:, 0]*train_percent), 0)
    validate_data = np.round(len(data[:, 0] * (1-train_percent)), 0)
    train = np.zeros((train_data, len(data[0, :])))
    validate = np.zeros((validate_data, len(data[0, :])))
    return train, validate