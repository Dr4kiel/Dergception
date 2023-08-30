# Classe d'une couche de neurones

import numpy as np


class Layer:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(output_size, input_size + 1) * 2 - 1
        self.values = np.zeros(output_size)

    def __str__(self):
        return "Layer:\n" + str(self.weights) + "\n" + str(self.values) + "\n"

    def update_weights(self, next_layer):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                if self.values[i] == 1:
                    self.values[i] = 0.9999999999999999
                if self.values[i] == 0:
                    self.values[i] = 0.0000000000000001
                # if next_layer is an array, then it's the output layer
                if not isinstance(next_layer, Layer):
                    self.weights[i][j] += 0.1 * self.values[i] * \
                        (1 - self.values[i]) * (next_layer[i] - self.values[i])
                # else, it's the next layer
                else:
                    # protect against out of bounds
                    if j < len(next_layer.weights):
                        self.weights[i][j] += 0.1 * self.values[i] * \
                            (1 - self.values[i]) * \
                            (next_layer.values[j] - self.values[i])

    def get_values(self):
        return self.values

    def set_values(self, previous_layer):
        # if previous_layer is an array, then it's the input layer
        if isinstance(previous_layer, np.ndarray):
            self.values = np.dot(self.weights, np.append(previous_layer, 1))
        # else, it's the previous layer
        else:
            self.values = np.dot(self.weights, np.append(
                previous_layer.values, 1))
        self.values = 1 / (1 + np.exp(-self.values))
